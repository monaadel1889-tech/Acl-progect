import io, os
from datetime import datetime
import numpy as np
import pandas as pd
import pdfplumber
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ---------- قاعدة البيانات ----------
# الافتراضي SQLite. للإنتاج استخدم PostgreSQL:
# export DATABASE_URL=postgresql+psycopg2://user:pass@host/dbname
DB_URL = os.getenv("DATABASE_URL", "sqlite:///fraud.db")
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    created = Column(DateTime, default=datetime.utcnow)
    rows = Column(Integer)
    flagged = Column(Integer)
    note = Column(Text, default="")
    items = relationship("Item", cascade="all, delete")


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    row_no = Column(Integer)
    date = Column(String(30))
    vendor = Column(String(255))
    invoice = Column(String(100))
    amount = Column(Float)
    score = Column(Float)
    reasons = Column(Text)


Base.metadata.create_all(engine)

# ---------- قراءة الملفات ----------
ALIASES = {
    "date": ["date", "تاريخ"],
    "amount": ["amount", "total", "value", "مبلغ", "قيمة", "اجمالي", "إجمالي"],
    "vendor": ["vendor", "supplier", "name", "مورد", "عميل", "اسم"],
    "invoice": ["invoice", "ref", "فاتورة", "رقم"],
}
THRESHOLDS = [float(x) for x in os.getenv("APPROVAL_LIMITS", "10000,50000").split(",")]
WEEKEND = {4, 5}  # الجمعة والسبت


def read_file(name: str, data: bytes) -> pd.DataFrame:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(data))
    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(data))
    elif ext == "pdf":
        rows = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                for t in page.extract_tables():
                    rows += t
        if len(rows) < 2:
            raise HTTPException(400, "لم يتم العثور على جدول داخل ملف PDF (الملفات الممسوحة ضوئيا تحتاج OCR).")
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        raise HTTPException(400, "الصيغ المدعومة: xlsx, xls, csv, pdf")

    mapping = {}
    for col in df.columns:
        c = str(col).strip().lower()
        for key, words in ALIASES.items():
            if key not in mapping.values() and any(w in c for w in words):
                mapping[col] = key
                break
    df = df.rename(columns=mapping)
    if "amount" not in df.columns:
        raise HTTPException(400, "لم أجد عمود المبلغ (amount / المبلغ).")
    for k in ("date", "vendor", "invoice"):
        if k not in df.columns:
            df[k] = None
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["amount"]).reset_index(drop=True)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# ---------- كشف الاحتيال ----------
def analyze(df: pd.DataFrame):
    n = len(df)
    score = np.zeros(n)
    why = [[] for _ in range(n)]

    def flag(mask, pts, text):
        for i in np.where(mask)[0]:
            score[i] += pts
            why[i].append(text)

    # 1) فواتير مكررة
    if df["invoice"].notna().any():
        flag(df.duplicated(["vendor", "invoice"], keep=False).values, 40, "فاتورة مكررة (نفس المورد والرقم)")
    flag(df.duplicated(["vendor", "amount", "date"], keep=False).values, 30, "نفس المورد والمبلغ والتاريخ")

    # 2) مبالغ مستديرة
    a = df["amount"].values
    flag((a >= 1000) & (a % 1000 == 0), 15, "مبلغ مستدير بشكل مشبوه")

    # 3) مبالغ قريبة تحت حد الموافقة
    for t in THRESHOLDS:
        flag((a >= t * 0.95) & (a < t), 25, f"أقل بقليل من حد الموافقة ({t:,.0f})")

    # 4) عمليات في العطلة + مبالغ صفرية/سالبة
    flag(df["date_parsed"].dt.weekday.isin(WEEKEND).values, 10, "عملية في يوم عطلة")
    flag(a <= 0, 20, "مبلغ صفر أو سالب")

    # 5) قانون بنفورد (على الملف كله)
    note = ""
    digits = pd.Series(np.abs(a)).astype(str).str.lstrip("0.").str[0]
    digits = pd.to_numeric(digits, errors="coerce")
    if n >= 100:
        obs = np.array([(digits == d).sum() for d in range(1, 10)], dtype=float)
        exp = np.log10(1 + 1 / np.arange(1, 10)) * obs.sum()
        p = chisquare(obs, exp).pvalue
        if p < 0.01:
            note = f"توزيع الأرقام الأولى لا يطابق قانون بنفورد (p={p:.4f}): قد تكون الأرقام مفبركة."
            over = [d for d in range(1, 10) if obs[d - 1] / obs.sum() - exp[d - 1] / obs.sum() > 0.05]
            flag(digits.isin(over).values, 10, "رقم أول مبالغ في تكراره (بنفورد)")

    # 6) نموذج ذكاء اصطناعي: Isolation Forest للقيم الشاذة
    if n >= 20:
        X = np.log1p(np.abs(a)).reshape(-1, 1)
        pred = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
        flag(pred == -1, 30, "قيمة شاذة إحصائيا (نموذج AI)")

    df["score"] = np.minimum(score, 100)
    df["reasons"] = ["، ".join(r) for r in why]
    return df, note


# ---------- الواجهة البرمجية ----------
app = FastAPI(title="Fraud Detector")


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "الملف أكبر من 25MB")
    df, note = analyze(read_file(file.filename, data))
    with Session() as s:
        rep = Report(filename=file.filename, rows=len(df), flagged=int((df["score"] >= 40).sum()), note=note)
        s.add(rep)
        s.flush()
        for i, r in df.iterrows():
            s.add(Item(report_id=rep.id, row_no=i + 1, date=str(r["date"]), vendor=str(r["vendor"]),
                       invoice=str(r["invoice"]), amount=float(r["amount"]),
                       score=float(r["score"]), reasons=r["reasons"]))
        s.commit()
        return {"report_id": rep.id}


@app.get("/reports")
def reports():
    with Session() as s:
        rs = s.query(Report).order_by(Report.id.desc()).limit(50).all()
        return [{"id": r.id, "filename": r.filename, "created": r.created.isoformat(),
                 "rows": r.rows, "flagged": r.flagged} for r in rs]


@app.get("/reports/{rid}")
def report(rid: int):
    with Session() as s:
        r = s.get(Report, rid)
        if not r:
            raise HTTPException(404, "التقرير غير موجود")
        items = sorted(r.items, key=lambda i: -i.score)[:500]
        return {"id": r.id, "filename": r.filename, "rows": r.rows, "flagged": r.flagged, "note": r.note,
                "items": [{"row": i.row_no, "date": i.date, "vendor": i.vendor, "invoice": i.invoice,
                           "amount": i.amount, "score": i.score, "reasons": i.reasons} for i in items]}
