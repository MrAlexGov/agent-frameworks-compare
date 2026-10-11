"""Общее для всех реализаций: база знаний, инструменты, задачи, проверка ответа.

База знаний — вымышленный облачный провайдер «Облачко», чтобы модель не могла
ответить из памяти и была вынуждена звать search_kb. Часть задач требует ещё
и calc (многошаговое рассуждение: найти цифры -> посчитать).
"""
import ast, operator, os, re
from typing import List

from pydantic import BaseModel, Field

KB = {
    "tariffs": "Тарифы «Облачко»: Старт — 990 ₽/мес (2 vCPU, 4 ГБ RAM, 50 ГБ SSD); Про — 2490 ₽/мес (4 vCPU, 8 ГБ RAM, "
               "100 ГБ SSD); Бизнес — 5900 ₽/мес (8 vCPU, 32 ГБ RAM, 400 ГБ SSD). При оплате за 12 месяцев скидка 15%, за 6 месяцев — 7%.",
    "gpu": "GPU-серверы «Облачко»: NVIDIA T4 — 38 ₽/час, NVIDIA L4 — 61 ₽/час, NVIDIA A100 80GB — 289 ₽/час. Минимальная аренда — 1 час, "
           "тарификация поминутная после первого часа.",
    "sla": "SLA «Облачко»: доступность 99,95% для тарифов Про и Бизнес, 99,5% для Старт. Компенсация — 10% месячной стоимости за каждые "
           "начатые 0,1% ниже SLA, но не более 50%.",
    "backup": "Резервное копирование: ежедневные снапшоты хранятся 7 дней бесплатно; хранение 30 дней стоит 3 ₽ за ГБ в месяц. "
              "Восстановление из снапшота занимает до 15 минут.",
    "regions": "Регионы: Москва (msk-1, msk-2), Санкт-Петербург (spb-1), Новосибирск (nsk-1). GPU A100 доступны только в msk-2.",
    "support": "Поддержка: Старт — тикеты, ответ до 24 часов; Про — тикеты и чат, ответ до 4 часов; Бизнес — выделенный менеджер, "
               "ответ до 30 минут, телефон 24/7.",
    "s3": "Объектное хранилище S3: 1,9 ₽ за ГБ в месяц, исходящий трафик — 1 ₽ за ГБ сверх 100 ГБ в месяц, входящий бесплатно.",
}


def search_kb(query: str) -> str:
    """Поиск по базе знаний «Облачко»: возвращает до 2 самых релевантных разделов."""
    import math
    stems = lambda t: {w[:5] for w in re.findall(r"\w+", t.lower()) if len(w) >= 2}  # грубый стемминг: префикс 5 букв
    docs = {k: stems(v + " " + k) for k, v in KB.items()}
    idf = lambda w: math.log((1 + len(docs)) / (1 + sum(w in d for d in docs.values())))  # редкие слова весят больше
    q = stems(query)
    scored = sorted(KB.items(), key=lambda kv: -sum(idf(w) for w in q & docs[kv[0]]))
    return "\n\n".join(f"[{k}] {v}" for k, v in scored[:2])


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def calc(expression: str) -> str:
    """Безопасный калькулятор: + - * / ** % и скобки."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    try:
        v = ev(ast.parse(expression.replace(",", ".").replace("×", "*"), mode="eval").body)
        return str(round(v, 4))
    except Exception as e:
        return f"error: {e}"


class Answer(BaseModel):
    answer: str = Field(description="Короткий окончательный ответ на русском")
    value: float | None = Field(default=None, description="Числовой ответ, если вопрос про число/сумму, иначе null")
    sources: List[str] = Field(default_factory=list, description="Ключи разделов базы знаний, например ['tariffs']")


SYSTEM = ("Ты ассистент облачного провайдера «Облачко». Факты бери ТОЛЬКО из базы знаний через search_kb, "
          "любые вычисления делай через calc. Если в базе нет ответа — так и скажи.")

# (вопрос, ожидаемое число или None, обязательные regex в ответе, минимум вызовов инструментов)
TASKS = [
    ("Сколько стоит тариф Про в месяц?", 2490, [], 1),
    ("Сколько vCPU на тарифе Бизнес?", 8, [], 1),
    ("Какая доступность по SLA у тарифа Старт?", 99.5, [], 1),
    ("В каком регионе доступны A100?", None, ["msk-2"], 1),
    ("За сколько отвечает поддержка на тарифе Про?", 4, [], 1),
    ("Сколько стоит год тарифа Про при оплате сразу за 12 месяцев?", 25398, [], 2),
    ("Сколько стоит полгода тарифа Бизнес при оплате за 6 месяцев?", 32922, [], 2),
    ("Сколько будет стоить 10 часов аренды A100?", 2890, [], 2),
    ("Сколько стоит хранить 500 ГБ в S3 месяц?", 950, [], 2),
    ("Сколько стоит отдать 300 ГБ исходящего трафика из S3 за месяц (только трафик)?", 200, [], 2),
    ("Хранение 30 дней снапшотов объёмом 200 ГБ — сколько в месяц?", 600, [], 2),
    ("Насколько дороже час L4, чем час T4, в рублях?", 23, [], 2),
    ("Доступность на тарифе Бизнес упала до 99,7%. Какая компенсация в рублях?", 1770, [], 2),
    ("Можно ли арендовать GPU H100?", None, ["нет|недоступ|не предоставл"], 1),
    ("Сколько стоит тариф Старт и Про вместе в месяц?", 3480, [], 2),
    ("Есть ли у Облачко регион в Казани?", None, ["нет|отсутств"], 1),
]


def check(task, out: Answer) -> bool:
    q, num, subs, _ = task
    text = (out.answer or "").lower().replace(" ", " ")
    ok = True
    if num is not None:
        nums = [float(x.replace(" ", "").replace(",", ".")) for x in re.findall(r"\d[\d ]*(?:[.,]\d+)?", text)]
        cand = [out.value] if out.value is not None else nums  # value приоритетнее текста
        ok = any(abs(c - num) <= max(0.5, abs(num) * 0.001) for c in cand)
    return ok and all(re.search(s, text) for s in subs)


def model_cfg():
    return dict(base_url=os.environ["EVAL_BASE_URL"], api_key=os.environ["EVAL_API_KEY"])
