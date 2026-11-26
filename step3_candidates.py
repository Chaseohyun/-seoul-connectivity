# step3_candidates.py: 신규 녹지 1 ha 후보지 선정 및 위치별 연결성 기여도 산정 스크립트
# 규칙 준수: 한국어 주석 작성, 기준값 하드코딩 배제, 지정된 8개 라이브러리만 활용

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
import rasterio
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    total_start_time = time.time()
    print("=== [프롬프트 3] 신규 녹지 1 ha 위치별 연결성 기여도 산정 시작 ===")

    # 1. 파일 경로 설정
    input_tif = "seoul_s2_20260516.tif"
    input_ndvi_tif = "seoul_ndvi.tif"
    boundary_gpkg = "seoul_boundary.gpkg"
    conn_gpkg = "patch_connectivity.gpkg"
    links_csv = "patch_links.csv"
    pstar_npy = "pstar_top2000.npy"

    output_candidates_gpkg = "candidates.gpkg"
    output_summary = "step3_summary.txt"
    output_map = "step3_map.png"

    # 2. 기존 패치 및 연결성 데이터 읽기
    print("\n1. 기존 패치 및 네트워크 데이터 로드...")
    gdf_patches = gpd.read_file(conn_gpkg, layer="patches", engine="pyogrio")
    gdf_patches = gdf_patches.sort_values("pid").reset_index(drop=True)
    N = len(gdf_patches)
    patch_geoms = gdf_patches.geometry.values
    a = gdf_patches["ha"].values.astype(np.float64)

    # 기준 Q 계산: Σ_i a_i * (flux_ha_i + a_i)
    Q_base = float(np.sum(a * (gdf_patches["flux_ha"].values + a)))
    print(f"  - 기존 패치 수: {N:,}개, 기준 Q: {Q_base:,.0f} ha²")

    # 상위 2000개 패치 간 p* 행렬 및 링크 로드
    pstar_top2000 = np.load(pstar_npy)
    df_links = pd.read_csv(links_csv)

    # 3. 경계 벡터 및 래스터 데이터 로드
    print("\n2. 서울 경계 및 위성/NDVI 래스터 확인...")
    seoul_gdf = gpd.read_file(boundary_gpkg, layer="seoul", engine="pyogrio")
    seoul_poly = seoul_gdf.geometry.iloc[0]

    gu_gdf = gpd.read_file(boundary_gpkg, layer="gu", engine="pyogrio")

    with rasterio.open(input_ndvi_tif) as src_ndvi:
        ndvi_array = src_ndvi.read(1)
        transform = src_ndvi.transform

    with rasterio.open(input_tif) as src_s2:
        b2_array = src_s2.read(2)  # 밴드 2: B08 근적외 (수면 판별: < 800)

    # 4. 후보지 격자 생성 및 3가지 조건 필터링
    print("\n3. 후보지 격자(500m 배수 점) 생성 및 적합성 필터링...")
    minx, miny, maxx, maxy = seoul_poly.bounds
    xs = np.arange(int(np.floor(minx / 500.0) * 500), int(np.ceil(maxx / 500.0) * 500) + 1, 500)
    ys = np.arange(int(np.floor(miny / 500.0) * 500), int(np.ceil(maxy / 500.0) * 500) + 1, 500)

    inv_transform = ~transform
    inside_seoul_count = 0
    valid_squares = []
    valid_coords = []

    for y in ys:
        for x in xs:
            # 중심 (x, y), 크기 100m x 100m (1 ha)
            sq = box(x - 50.0, y - 50.0, x + 50.0, y + 50.0)

            # 조건 ①: 정사각형 전체가 서울 경계 내부에 포함
            if not seoul_poly.contains(sq):
                continue
            inside_seoul_count += 1

            # 래스터 윈도우 인덱스 변환
            c_min, r_max = inv_transform * (x - 50.0, y - 50.0)
            c_max, r_min = inv_transform * (x + 50.0, y + 50.0)
            c0, c1 = int(round(c_min)), int(round(c_max))
            r0, r1 = int(round(r_min)), int(round(r_max))

            sq_ndvi = ndvi_array[r0:r1, c0:c1]
            sq_b2 = b2_array[r0:r1, c0:c1]

            # 조건 ②: 정사각형 내 녹지 화소(NDVI >= 0.6) 전무
            if np.any(sq_ndvi >= 0.6):
                continue

            # 조건 ③: 정사각형 내 수면 화소(B08 < 800) 전무
            if np.any(sq_b2 < 800):
                continue

            valid_squares.append(sq)
            valid_coords.append((x, y))

    print(f"  - 서울 경계 내 후보 격자(조건 ①): {inside_seoul_count:,}개")
    print(f"  - 세 조건을 모두 만족하는 최종 후보지: {len(valid_squares):,}개")

    # 후보지 데이터프레임 구성 및 정렬 규칙 적용: 남->북(y 오름차순), 서->동(x 오름차순)
    df_cand = pd.DataFrame(valid_coords, columns=["x", "y"])
    df_cand["geometry"] = valid_squares
    df_cand = df_cand.sort_values(by=["y", "x"], ascending=[True, True]).reset_index(drop=True)
    df_cand["cid"] = np.arange(1, len(df_cand) + 1, dtype=int)
    C = len(df_cand)

    # 중심점이 속한 자치구(gu_nm) 공간 조인
    cand_centers = gpd.points_from_xy(df_cand["x"], df_cand["y"], crs="EPSG:5186")
    pt_gdf = gpd.GeoDataFrame(df_cand[["cid"]], geometry=cand_centers, crs="EPSG:5186")
    joined_gu = gpd.sjoin(pt_gdf, gu_gdf[["gu_nm", "geometry"]], how="left", predicate="within")
    df_cand["gu_nm"] = joined_gu.sort_values("cid")["gu_nm"].values

    # 5. 후보지와 기존 패치 간 직접 링크 탐색 (500m 이내)
    print("\n4. 후보지-패치 직접 링크 탐색 (가장자리 간 거리 <= 500m)...")
    cand_geoms = df_cand.geometry.values
    patch_tree = shapely.STRtree(patch_geoms)

    q_res = patch_tree.query(cand_geoms, predicate="dwithin", distance=500.0)
    cand_idx_arr = q_res[0]
    patch_idx_arr = q_res[1]

    cp_dists = shapely.distance(cand_geoms[cand_idx_arr], patch_geoms[patch_idx_arr])
    valid_cp_mask = cp_dists <= 500.0
    cand_idx_arr = cand_idx_arr[valid_cp_mask]
    patch_idx_arr = patch_idx_arr[valid_cp_mask]
    cp_dists = cp_dists[valid_cp_mask]

    # 각 후보지별 직접 링크 수(n_link) 및 최단 거리(nearest_m) 산출
    n_link = np.zeros(C, dtype=int)
    nearest_m = np.full(C, np.inf, dtype=np.float64)

    for c_i, d_val in zip(cand_idx_arr, cp_dists):
        n_link[c_i] += 1
        if d_val < nearest_m[c_i]:
            nearest_m[c_i] = d_val

    df_cand["n_link"] = n_link
    df_cand["nearest_m"] = nearest_m
    no_link_cands = int(np.count_nonzero(n_link == 0))
    print(f"  - 생성된 후보지-패치 링크: {len(cand_idx_arr):,}쌍 (링크 없는 후보지: {no_link_cands}개)")

    # 6. 방향성 그래프 구축 및 다익스트라 최단 경로 계산
    # 패치 간 양방향 링크 + 후보지->패치 단방향 링크
    print("\n5. 방향성 네트워크 구축 및 후보지별 최단 경로(p*) 산정...")
    p_a = df_links["pid_a"].values - 1
    p_b = df_links["pid_b"].values - 1
    p_d = df_links["dist_m"].values

    cp_from = N + cand_idx_arr  # 후보지 노드 인덱스: N .. N+C-1
    cp_to = patch_idx_arr       # 패치 노드 인덱스: 0 .. N-1

    row_indices = np.concatenate([p_a, p_b, cp_from])
    col_indices = np.concatenate([p_b, p_a, cp_to])
    data_weights = np.concatenate([p_d, p_d, cp_dists])

    total_nodes = N + C
    graph_csr = csr_matrix((data_weights, (row_indices, col_indices)), shape=(total_nodes, total_nodes))

    # 후보지 노드들로부터 패치들로의 다익스트라 최단 경로
    decay = -np.log(2.0) / 250.0
    cand_nodes = np.arange(N, N + C)
    dist_matrix = dijkstra(graph_csr, directed=True, indices=cand_nodes)

    # 후보지에서 기존 패치들(0..N-1)로의 거리 및 p*
    D_cand_to_patches = dist_matrix[:, :N]
    P_star = np.exp(decay * D_cand_to_patches)
    P_star[np.isneginf(decay * D_cand_to_patches)] = 0.0

    # 7. Q 증가량(dQ) 3개 성분 계산
    print("\n6. 연결성 증가량 (dQ_intra, dQ_flux, dQ_conn) 계산...")
    # 1) dQ_intra = a_c^2 = 1.0
    dQ_intra = np.ones(C, dtype=np.float64)

    # 2) dQ_flux = 2 * a_c * Σ_j a_j * p*_cj = 2 * (P_star @ a)
    cand_flux_ha = P_star @ a
    dQ_flux = 2.0 * cand_flux_ha

    # 3) dQ_conn = Σ_i Σ_{j≠i} a_i * a_j * max(0, p*_ci * p*_cj - p*_ij) (상위 2000개 대상)
    a_top = a[:2000]
    A_mat = np.outer(a_top, a_top)
    P_top = P_star[:, :2000]

    dQ_conn = np.zeros(C, dtype=np.float64)
    for c in range(C):
        p_vec = P_top[c]
        P_cand = np.outer(p_vec, p_vec)
        diff = np.maximum(0.0, P_cand - pstar_top2000)
        dQ_conn[c] = float(np.sum(A_mat * diff))

    # 종합 지표 집계
    dQ = dQ_intra + dQ_flux + dQ_conn
    dEC_ha = np.sqrt(Q_base + dQ) - np.sqrt(Q_base)
    dPC_add = 100.0 * dQ / Q_base
    conn_pct = 100.0 * dQ_conn / dQ

    df_cand["flux_ha"] = cand_flux_ha
    df_cand["dQ_intra"] = dQ_intra
    df_cand["dQ_flux"] = dQ_flux
    df_cand["dQ_conn"] = dQ_conn
    df_cand["dQ"] = dQ
    df_cand["dPC_add"] = dPC_add
    df_cand["dEC_ha"] = dEC_ha
    df_cand["conn_pct"] = conn_pct
    df_cand["rank"] = df_cand["dEC_ha"].rank(ascending=False, method="min").astype(int)

    # candidates.gpkg 저장
    print("\n7. 후보지 평가 결과 벡터 저장 (candidates.gpkg)...")
    field_order = [
        "cid", "gu_nm", "x", "y", "n_link", "nearest_m", "flux_ha",
        "dQ_intra", "dQ_flux", "dQ_conn", "dQ", "dPC_add", "dEC_ha",
        "conn_pct", "rank", "geometry"
    ]
    gdf_candidates = gpd.GeoDataFrame(df_cand[field_order], geometry="geometry", crs="EPSG:5186")
    gdf_candidates.to_file(output_candidates_gpkg, layer="candidates", driver="GPKG", engine="pyogrio")
    print(f"  - {output_candidates_gpkg} (레이어: candidates) 저장 완료.")

    # 8. 요약 통계 집계 및 작성
    print("\n8. 통계 분석 및 요약문 작성 (step3_summary.txt)...")
    dec_min = float(df_cand["dEC_ha"].min())
    dec_median = float(np.median(df_cand["dEC_ha"]))
    dec_mean = float(df_cand["dEC_ha"].mean())
    dec_max = float(df_cand["dEC_ha"].max())

    # 상위 10개 후보지
    top10_df = df_cand.sort_values("rank").head(10)
    top10_lines = []
    for _, row in top10_df.iterrows():
        top10_lines.append(
            f"  순위 {int(row['rank'])}위 (cid {int(row['cid'])}): {row['gu_nm']} ({int(row['x'])}, {int(row['y'])}) "
            f"| dEC: {row['dEC_ha']:.2f} ha, 경유 기여: {row['conn_pct']:.0f}%, 링크: {int(row['n_link'])}개"
        )

    # 자치구별 평균
    gu_summary = df_cand.groupby("gu_nm").agg(
        후보지수=("cid", "count"),
        dEC_평균=("dEC_ha", "mean"),
        dEC_최대=("dEC_ha", "max")
    ).sort_values("dEC_평균", ascending=False)

    gu_lines = []
    for gu, r in gu_summary.iterrows():
        gu_lines.append(f"  - {gu:6s}: 후보지 {r['후보지수']:2.0f}곳 | 평균 dEC {r['dEC_평균']:.4f} ha | 최대 dEC {r['dEC_최대']:.2f} ha")

    summary_content = f"""[프롬프트 3] 신규 녹지 1 ha 위치별 연결성 기여 분석 결과 요약
--------------------------------------------------------------
- 서울 경계 내 500m 격자: {inside_seoul_count:,}개
- 3개 조건(경계 내부, 녹지 없음, 수면 없음) 만족 후보지: {C:,}곳
- 직접 링크 없는 후보지 수: {no_link_cands}개
- dEC(ha) 통계: 최소 {dec_min:.4f}, 중앙값 {dec_median:.4f}, 평균 {dec_mean:.4f}, 최대 {dec_max:.2f} ha

[상위 10개 후보지 목록]
{chr(10).join(top10_lines)}

[자치구별 후보지 연결성 기여 평균 (내림차순)]
{chr(10).join(gu_lines)}
--------------------------------------------------------------
"""
    with open(output_summary, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"  - {output_summary} 저장 완료.")

    # 9. 지도 시각화 (step3_map.png)
    print("\n9. 시각화 지도 생성 (step3_map.png)...")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # 서울 자치구 경계 플롯
    gu_gdf.plot(ax=ax, facecolor="#f8f9fa", edgecolor="#b0bec5", linewidth=0.7)

    # 기존 녹지 패치 배경 플롯 (연한 녹색)
    gdf_patches.plot(ax=ax, color="#81c784", alpha=0.5, linewidth=0)

    # 후보지 dEC_ha 색칠 플롯 (로그 눈금)
    norm = mcolors.LogNorm(vmin=dec_min, vmax=dec_max)
    gdf_candidates.plot(
        ax=ax,
        column="dEC_ha",
        cmap="YlOrRd",
        norm=norm,
        legend=True,
        legend_kwds={
            "label": "신규 녹지 1 ha 추가에 따른 dEC (ha, 로그 눈금)",
            "orientation": "horizontal",
            "fraction": 0.035,
            "pad": 0.06
        }
    )

    # 상위 10개 후보지 순위 표시
    for _, row in top10_df.iterrows():
        rx, ry = row["x"], row["y"]
        rk = int(row["rank"])
        ax.scatter([rx], [ry], color="#d32f2f", s=80, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(
            str(rk),
            (rx, ry),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#b71c1c",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="#d32f2f", alpha=0.9),
            zorder=6
        )

    ax.set_title("서울 신규 녹지 1 ha 후보지 위치별 연결성 기여도 (dEC)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("X 좌표 (m, EPSG:5186)")
    ax.set_ylabel("Y 좌표 (m, EPSG:5186)")

    plt.tight_layout()
    plt.savefig(output_map, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  - {output_map} 저장 완료.")

    total_elapsed = time.time() - total_start_time
    print(f"\n=== [프롬프트 3] 전체 작업 완료 (총 소요 시간: {total_elapsed:.1f}초) ===")


if __name__ == "__main__":
    main()
