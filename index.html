"""
선형회귀 / 비선형회귀 계산 프로그램 (외부 라이브러리 없이 직접 구현)

구현 내용
1. 선형회귀        : y = a + b*x               (최소제곱법 공식)
2. 다항식 회귀     : y = c0 + c1*x + ... + cn*x^n (정규방정식 + 가우스 소거법)
3. 지수 회귀       : y = a * e^(b*x)           (로그 변환 후 선형회귀)
4. 거듭제곱 회귀   : y = a * x^b               (양변 로그 변환 후 선형회귀)
5. 결정계수 R^2    : 모델이 데이터를 얼마나 잘 설명하는지 평가

예제 데이터: 선박 속력(knot)에 따른 하루 연료 소모량(톤/일)
  - 선박의 추진 동력은 대략 속력의 세제곱에 비례(애드미럴티 공식)하므로
    연료 소모량은 속력에 대해 비선형적으로 증가한다.
"""

import math


# ---------------------------------------------------------------
# 공통 도구
# ---------------------------------------------------------------
def mean(values):
    return sum(values) / len(values)


def r_squared(y, y_pred):
    """결정계수 R^2 = 1 - SSE/SST"""
    y_mean = mean(y)
    sse = sum((yi - pi) ** 2 for yi, pi in zip(y, y_pred))  # 잔차 제곱합
    sst = sum((yi - y_mean) ** 2 for yi in y)                # 총 제곱합
    return 1 - sse / sst if sst != 0 else 1.0


def solve_linear_system(A, b):
    """가우스 소거법(부분 피벗팅)으로 Ax = b 를 푼다."""
    n = len(A)
    # 확장 행렬 [A | b] 생성 (원본 보존을 위해 복사)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # 전진 소거
    for col in range(n):
        # 절댓값이 가장 큰 행을 피벗으로 선택 (수치 안정성)
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("행렬이 특이(singular)하여 해를 구할 수 없습니다.")
        M[col], M[pivot] = M[pivot], M[col]

        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    # 후진 대입
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


# ---------------------------------------------------------------
# 1. 선형회귀
# ---------------------------------------------------------------
def linear_regression(x, y):
    """y = a + b*x 의 절편 a, 기울기 b 반환"""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x, y의 길이가 같고 2개 이상이어야 합니다.")
    x_mean, y_mean = mean(x), mean(y)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    sxx = sum((xi - x_mean) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError("x 값이 모두 같아 기울기를 구할 수 없습니다.")
    b = sxy / sxx
    a = y_mean - b * x_mean
    return a, b


# ---------------------------------------------------------------
# 2. 다항식 회귀 (비선형)
# ---------------------------------------------------------------
def polynomial_regression(x, y, degree):
    """
    정규방정식 (X^T X) c = X^T y 를 풀어 계수 [c0, c1, ..., cn] 반환.
    X^T X 의 (i, j) 원소 = sum(x^(i+j)),  X^T y 의 i 원소 = sum(x^i * y)
    """
    if len(x) <= degree:
        raise ValueError("데이터 개수가 차수보다 많아야 합니다.")
    n = degree + 1
    A = [[sum(xi ** (i + j) for xi in x) for j in range(n)] for i in range(n)]
    B = [sum((xi ** i) * yi for xi, yi in zip(x, y)) for i in range(n)]
    return solve_linear_system(A, B)


def poly_predict(coeffs, xv):
    return sum(c * xv ** i for i, c in enumerate(coeffs))


# ---------------------------------------------------------------
# 3. 지수 회귀 (비선형)
# ---------------------------------------------------------------
def exponential_regression(x, y):
    """
    y = a * e^(b*x)  →  ln(y) = ln(a) + b*x  로 바꿔 선형회귀 적용.
    (y > 0 이어야 함)
    """
    if any(yi <= 0 for yi in y):
        raise ValueError("지수 회귀는 y > 0 인 데이터에만 사용할 수 있습니다.")
    ln_y = [math.log(yi) for yi in y]
    ln_a, b = linear_regression(x, ln_y)
    return math.exp(ln_a), b


# ---------------------------------------------------------------
# 4. 거듭제곱 회귀 (비선형)
# ---------------------------------------------------------------
def power_regression(x, y):
    """
    y = a * x^b  →  ln(y) = ln(a) + b*ln(x)  로 바꿔 선형회귀 적용.
    (x > 0, y > 0 이어야 함)  b ≈ 3 이면 '연료 ∝ 속력^3' 관계와 일치.
    """
    if any(v <= 0 for v in x) or any(v <= 0 for v in y):
        raise ValueError("거듭제곱 회귀는 x > 0, y > 0 인 데이터에만 사용할 수 있습니다.")
    ln_x = [math.log(v) for v in x]
    ln_y = [math.log(v) for v in y]
    ln_a, b = linear_regression(ln_x, ln_y)
    return math.exp(ln_a), b


# ---------------------------------------------------------------
# 5. 그래프 그리기 (라이브러리 없이 SVG 파일을 직접 생성)
# ---------------------------------------------------------------
def nice_ticks(lo, hi, count=6):
    """축 눈금을 보기 좋은 간격(1, 2, 5 x 10^k)으로 계산"""
    span = hi - lo if hi > lo else 1
    raw = span / count
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 5, 10) if s * mag >= raw))
    first = math.floor(lo / step)
    last = math.ceil(hi / step)
    return [round(k * step, 10) for k in range(first, last + 1)]


def draw_svg(x, y, models, x_label="x", y_label="y", filename="regression_graph.svg"):
    """
    models: [(이름, 예측함수, 색상), ...]
    데이터 점과 각 회귀 곡선을 SVG로 저장한다.
    """
    W, H = 760, 500
    title = f"{y_label} vs {x_label}"
    left, right, top, bottom = 70, 200, 50, 60  # 여백
    pw, ph = W - left - right, H - top - bottom

    # x축 범위를 정하고, 곡선을 그릴 촘촘한 x 값 생성
    xt = nice_ticks(min(x), max(x))
    xs = [xt[0] + (xt[-1] - xt[0]) * i / 200 for i in range(201)]

    curves = [(name, [f(v) for v in xs], color) for name, f, color in models]
    all_y = list(y) + [v for _, ys, _ in curves for v in ys if math.isfinite(v)]
    yt = nice_ticks(min(all_y), max(all_y))
    X0, X1, Y0, Y1 = xt[0], xt[-1], yt[0], yt[-1]

    def sx(v):  # 데이터 좌표 → 화면 좌표
        return left + (v - X0) / (X1 - X0) * pw

    def sy(v):
        return top + ph - (v - Y0) / (Y1 - Y0) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{left + pw / 2}" y="28" font-size="18" text-anchor="middle" '
           f'font-weight="bold">{title}</text>']

    # 격자와 눈금
    for t in xt:
        out.append(f'<line x1="{sx(t):.1f}" y1="{top}" x2="{sx(t):.1f}" y2="{top + ph}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{sx(t):.1f}" y="{top + ph + 20}" font-size="12" text-anchor="middle">{t:g}</text>')
    for t in yt:
        out.append(f'<line x1="{left}" y1="{sy(t):.1f}" x2="{left + pw}" y2="{sy(t):.1f}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{left - 8}" y="{sy(t) + 4:.1f}" font-size="12" text-anchor="end">{t:g}</text>')
    out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#333"/>')
    out.append(f'<text x="{left + pw / 2}" y="{H - 15}" font-size="14" text-anchor="middle">{x_label}</text>')
    out.append(f'<text x="20" y="{top + ph / 2}" font-size="14" text-anchor="middle" '
               f'transform="rotate(-90 20 {top + ph / 2})">{y_label}</text>')

    # 그래프 영역 밖으로 나가는 선을 잘라내기
    out.append(f'<clipPath id="area"><rect x="{left}" y="{top}" width="{pw}" height="{ph}"/></clipPath>')

    # 회귀 곡선
    for name, ys, color in curves:
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, ys) if math.isfinite(b))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                   f'stroke-width="2.5" clip-path="url(#area)"/>')

    # 데이터 점
    for a, b in zip(x, y):
        out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="5" fill="#222"/>')

    # 범례
    lx, ly = left + pw + 20, top + 10
    out.append(f'<circle cx="{lx + 12}" cy="{ly}" r="5" fill="#222"/>')
    out.append(f'<text x="{lx + 30}" y="{ly + 4}" font-size="13">데이터</text>')
    for i, (name, _, color) in enumerate(curves, start=1):
        yy = ly + i * 26
        out.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 24}" y2="{yy}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{lx + 30}" y="{yy + 4}" font-size="13">{name}</text>')

    out.append("</svg>")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return filename


# ---------------------------------------------------------------
# 입력 / 실행
# ---------------------------------------------------------------
def read_data():
    print("데이터를 입력하세요. (그냥 Enter를 누르면 선박 예제 데이터 사용)")
    sx = input("x 값 (공백으로 구분): ").strip()
    if not sx:
        # 선박 속력(knot)과 하루 연료 소모량(톤/일) 예제
        x = [10, 12, 14, 16, 18, 20, 22, 24]
        y = [6.8, 11.0, 18.3, 26.1, 38.6, 51.2, 70.1, 89.0]
        print("[선박 예제] 속력(knot)에 따른 연료 소모량(톤/일)")
        print(f"  속력      = {x}")
        print(f"  연료 소모량 = {y}")
        return x, y, "속력 (knot)", "연료 소모량 (톤/일)"

    sy = input("y 값 (공백으로 구분): ").strip()
    x = [float(v) for v in sx.split()]
    y = [float(v) for v in sy.split()]
    if len(x) != len(y):
        raise ValueError("x와 y의 개수가 같아야 합니다.")
    x_label = input("x축 이름 (Enter = x): ").strip() or "x"
    y_label = input("y축 이름 (Enter = y): ").strip() or "y"
    return x, y, x_label, y_label


def main():
    x, y, x_label, y_label = read_data()
    models = []  # 그래프에 그릴 (이름, 예측함수, 색상)
    print("\n" + "=" * 50)

    # 1. 선형회귀
    a, b = linear_regression(x, y)
    pred = [a + b * xi for xi in x]
    print("[1] 선형회귀")
    print(f"    y = {a:.4f} + {b:.4f}x")
    print(f"    R^2 = {r_squared(y, pred):.4f}")
    models.append(("선형", lambda v: a + b * v, "#1f77b4"))

    # 2. 다항식 회귀
    deg_in = input("\n다항식 차수를 입력하세요 (기본 2): ").strip()
    degree = int(deg_in) if deg_in else 2
    coeffs = polynomial_regression(x, y, degree)
    pred = [poly_predict(coeffs, xi) for xi in x]
    terms = " + ".join(
        f"{c:.4f}" if i == 0 else f"{c:.4f}x" if i == 1 else f"{c:.4f}x^{i}"
        for i, c in enumerate(coeffs)
    )
    print(f"[2] {degree}차 다항식 회귀")
    print(f"    y = {terms}")
    print(f"    R^2 = {r_squared(y, pred):.4f}")
    models.append((f"{degree}차 다항식", lambda v: poly_predict(coeffs, v), "#ff7f0e"))

    # 3. 지수 회귀
    print("\n[3] 지수 회귀")
    try:
        ea, eb = exponential_regression(x, y)
        pred = [ea * math.exp(eb * xi) for xi in x]
        print(f"    y = {ea:.4f} * e^({eb:.4f}x)")
        print(f"    R^2 = {r_squared(y, pred):.4f}")
        models.append(("지수", lambda v: ea * math.exp(eb * v), "#2ca02c"))
    except ValueError as e:
        print(f"    계산 불가: {e}")

    # 4. 거듭제곱 회귀
    print("\n[4] 거듭제곱 회귀")
    try:
        pa, pb = power_regression(x, y)
        pred = [pa * xi ** pb for xi in x]
        print(f"    y = {pa:.6f} * x^{pb:.4f}")
        print(f"    R^2 = {r_squared(y, pred):.4f}")
        models.append(("거듭제곱", lambda v: pa * v ** pb if v > 0 else float("nan"), "#d62728"))
    except ValueError as e:
        print(f"    계산 불가: {e}")

    print("=" * 50)

    # 5. 그래프 저장 후 브라우저로 열기
    path = draw_svg(x, y, models, x_label, y_label)
    print(f"\n그래프를 '{path}' 파일로 저장했습니다.")
    try:
        import os, webbrowser  # 파이썬 기본 내장 모듈
        webbrowser.open("file://" + os.path.abspath(path))
    except Exception:
        pass


if __name__ == "__main__":
    main()
