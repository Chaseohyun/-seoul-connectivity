# step5_species.py: 브랜치 실험 2 — 종별 분산거리(d50) 민감도 분석 스크립트
# 규칙 준수: 한국어 주석 작성, 기준값 하드코딩 배제, 지정된 8개 라이브러리만 활용

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def main():
    total_start_time = time.time()
    print("=== [프롬프트 5] 종별 분산거리(d50) 민감도 분석 시작 ===")

    # 1. 파일 경로 설정
    input_patches_gpkg = "green_patches.gpkg"
    input_candidates_gpkg = "candidates.gpkg"
    output_summary = "step5_summary.txt"
    output_candidates_csv = "step5_candidates.csv"
    output_curve_png = "step5_curve.png"

    # 2. 패치 데이터 로드 및 1 ha 이상 노드 필터링
    print("\n1. 패치 데이터 로드 및 면적 1 ha 이상 노드 필터링...")
    gdf_patches = gpd.read_file(input_patches_gpkg, layer="patches", engine="pyogrio")
    gdf_705 = gdf_patches[gdf_patches["ha"] >= 1.0].sort_values("pid").reset_index(drop=True)
    N = len(gdf_705)
    patch_geoms = gdf_705.geometry.values
    a = gdf_705["ha"].values.astype(np.float64)
    total_area_ha = float(np.sum(a))
    print(f"  - 분석 노드 수: {N:,}개 (면적 합: {total_area_ha:.1f} ha)")

    # 3. 705개 패치 간 모든 쌍(248,160개) 최단 거리 계산
    print("\n2. 패치 간 모든 쌍 최단 거리(m) 행렬 일괄 계산...")
    t_dist = time.time()
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = shapely.distance(patch_geoms[i_idx], patch_geoms[j_idx])
    num_pairs = len(pair_dists)
    print(f"  - 모든 쌍({num_pairs:,}개) 거리 계산 완료 소요 시간: {time.time() - t_dist:.1f}초")

    # 대칭 거리 행렬 구성
    D_edge = np.zeros((N, N), dtype=np.float64)
    D_edge[i_idx, j_idx] = pair_dists
    D_edge[j_idx, i_idx] = pair_dists

    # 4. 후보지 데이터 로드 및 후보지-패치 거리 계산
    print("\n3. 신규 녹지 후보지(candidates.gpkg) 로드 및 패치 간 거리 계산...")
    gdf_cands = gpd.read_file(input_candidates_gpkg, layer="candidates", engine="pyogrio")
    gdf_cands = gdf_cands.sort_values("cid").reset_index(drop=True)
    C = len(gdf_cands)
    cand_geoms = gdf_cands.geometry.values

    t_cdist = time.time()
    c_grid, p_grid = np.meshgrid(np.arange(C), np.arange(N), indexing="ij")
    cp_dist_mat = shapely.distance(cand_geoms[c_grid.ravel()], patch_geoms[p_grid.ravel()]).reshape(C, N)
    print(f"  - 후보지-패치 간 {C} × {N} 거리 행렬 계산 완료 소요 시간: {time.time() - t_cdist:.1f}초")

    # 5. 4개 대상 종 및 민감도 분석 경우 정의
    cases = [
        {"name": "맹꽁이", "d50": 100.0},
        {"name": "기준",   "d50": 250.0},
        {"name": "박새",   "d50": 650.0},
        {"name": "삵",     "d50": 11400.0}
    ]

    A_L = 60668.09
    A_mat = np.outer(a, a)

    results = []
    dEC_dict = {}
    top10_cids = {}

    print("\n4. 종별 연결성(EC) 및 후보지 기여도(dEC) 분석 수행...")
    for c_info in cases:
        name = c_info["name"]
        d50 = c_info["d50"]
        limit = 2.0 * d50
        print(f"\n  [{name}] d50 = {d50:,.0f} m, 직접 링크 한도 = {limit:,.0f} m 분석 중...")

        # 패치 간 유효 직접 링크
        valid_links = D_edge <= limit
        num_links = int(np.count_nonzero(valid_links[i_idx, j_idx]))

        # 연결 성분(컴포넌트) 수
        n_comp, _ = connected_components(valid_links, directed=False)

        # 다익스트라 최단 경로 연산
        D_graph = np.where(valid_links, D_edge, np.inf)
        np.fill_diagonal(D_graph, 0.0)

        decay = -np.log(2.0) / d50
        D_sp = dijkstra(D_graph, directed=False)
        P_star = np.exp(decay * D_sp)
        P_star[np.isneginf(decay * D_sp)] = 0.0
        np.fill_diagonal(P_star, 1.0)

        # Q 및 EC 산정
        Q = float(a @ P_star @ a)
        PC = Q / (A_L ** 2)
        EC = float(np.sqrt(Q))
        EC_pct = (EC / total_area_ha) * 100.0

        # pid 1과 pid 2 사이의 p* (0-indexed: index 0 and 1)
        p_1_2 = float(P_star[0, 1])

        # 후보지 기여도(dEC) 산정
        cp_valid = cp_dist_mat <= limit
        P_direct = np.where(cp_valid, np.exp(decay * cp_dist_mat), 0.0)

        no_link_count = 0
        dECs = np.zeros(C, dtype=np.float64)

        for c_idx in range(C):
            if not np.any(cp_valid[c_idx]):
                no_link_count += 1
                dQ = 1.0
            else:
                # p*_cj = max_k ( p_ck * p*_kj )
                p_vec = np.max(P_direct[c_idx, :, None] * P_star, axis=0)
                p_outer = np.outer(p_vec, p_vec)
                diff = np.maximum(0.0, p_outer - P_star)
                dQ = 1.0 + 2.0 * float(p_vec @ a) + float(np.sum(A_mat * diff))

            dECs[c_idx] = float(np.sqrt(Q + dQ) - np.sqrt(Q))

        dEC_dict[name] = dECs

        # 상위 후보지 추출
        sorted_indices = np.argsort(dECs)[::-1]
        top10_cids[name] = set(gdf_cands.iloc[sorted_indices[:10]]["cid"].values)

        top5_list = []
        for rk, s_idx in enumerate(sorted_indices[:5], 1):
            row = gdf_cands.iloc[s_idx]
            top5_list.append(
                f"      {rk}위: {row['gu_nm']} ({int(row['x'])}, {int(row['y'])}) - dEC: {dECs[s_idx]:.2f} ha"
            )

        results.append({
            "name": name,
            "d50": d50,
            "limit": limit,
            "links": num_links,
            "components": n_comp,
            "Q": Q,
            "PC": PC,
            "EC": EC,
            "EC_pct": EC_pct,
            "p_1_2": p_1_2,
            "no_link": no_link_count,
            "top5_lines": top5_list
        })

    # 6. step5_candidates.csv 저장
    print("\n5. 후보지별 종별 dEC 테이블 저장 (step5_candidates.csv)...")
    df_out_cands = pd.DataFrame({
        "cid": gdf_cands["cid"],
        "gu_nm": gdf_cands["gu_nm"],
        "x": gdf_cands["x"],
        "y": gdf_cands["y"],
        "dEC_맹꽁이": dEC_dict["맹꽁이"],
        "dEC_기준": dEC_dict["기준"],
        "dEC_박새": dEC_dict["박새"],
        "dEC_삵": dEC_dict["삵"]
    })
    df_out_cands.to_csv(output_candidates_csv, index=False, encoding="utf-8-sig")
    print(f"  - {output_candidates_csv} 저장 완료.")

    # 7. Spearman 상관 및 상위 10곳 중복 수 계산
    print("\n6. 종 간 상관관계 및 상위 10곳 중복 분석...")
    case_names = [c["name"] for c in cases]
    corr_lines = []
    overlap_lines = []

    for i in range(len(case_names)):
        for j in range(i + 1, len(case_names)):
            n1 = case_names[i]
            n2 = case_names[j]
            rho, pval = spearmanr(dEC_dict[n1], dEC_dict[n2])
            overlap = len(top10_cids[n1].intersection(top10_cids[n2]))
            corr_lines.append(f"  - {n1} vs {n2}: Spearman rho = {rho:.4f} (p-value = {pval:.1e})")
            overlap_lines.append(f"  - {n1} vs {n2}: {overlap}곳 중복")

    # 8. step5_summary.txt 작성
    print("\n7. 요약 보고서 작성 (step5_summary.txt)...")
    summary_lines = ["[프롬프트 5] 종별 분산거리 민감도 분석 결과 요약", "============================================================================="]

    for res in results:
        summary_lines.append(
            f"[{res['name']}] d50: {res['d50']:,.0f} m (직접 링크 한도: {res['limit']:,.0f} m)\n"
            f"  - 직접 링크 수: {res['links']:,}쌍 | 컴포넌트 수: {res['components']:,}개\n"
            f"  - Q: {res['Q']:,.0f} ha² | EC: {res['EC']:,.1f} ha (녹지 면적의 {res['EC_pct']:.1f} %)\n"
            f"  - pid 1과 pid 2 간 p*: {res['p_1_2']:.3f}\n"
            f"  - 직접 링크 없는 후보지 수: {res['no_link']:,}곳\n"
            f"  - dEC 상위 5개 후보지:\n" + "\n".join(res["top5_lines"])
        )
        summary_lines.append("-----------------------------------------------------------------------------")

    summary_lines.append("\n[경우 간 dEC 순위 상관관계 (Spearman rho)]")
    summary_lines.extend(corr_lines)
    summary_lines.append("\n[상위 10개 후보지 중복 수]")
    summary_lines.extend(overlap_lines)
    summary_lines.append("=============================================================================")

    with open(output_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  - {output_summary} 저장 완료.")

    # 9. d50-EC 민감도 곡선 생성 (step5_curve.png)
    print("\n8. d50-EC 반응 곡선 시각화 (step5_curve.png)...")
    curve_d50s = [50.0, 100.0, 250.0, 650.0, 1000.0, 2500.0, 5000.0, 11400.0, 20000.0]
    curve_ECs = []

    for d_val in curve_d50s:
        l_val = 2.0 * d_val
        v_links = D_edge <= l_val
        d_gr = np.where(v_links, D_edge, np.inf)
        np.fill_diagonal(d_gr, 0.0)
        dec = -np.log(2.0) / d_val
        d_sp = dijkstra(d_gr, directed=False)
        p_st = np.exp(dec * d_sp)
        p_st[np.isneginf(dec * d_sp)] = 0.0
        np.fill_diagonal(p_st, 1.0)
        q_val = float(a @ p_st @ a)
        curve_ECs.append(np.sqrt(q_val))

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(curve_d50s, curve_ECs, color="#1b5e20", linewidth=2.5, marker="o", markersize=5, label="d50-EC 반응 곡선")

    # 4가지 대상 종 강조 표시
    colors = {"맹꽁이": "#e65100", "기준": "#2e7d32", "박새": "#0277bd", "삵": "#6a1b9a"}
    offsets = {
        "맹꽁이": (10, -20),
        "기준":   (10, -15),
        "박새":   (10, 10),
        "삵":     (-80, 10)
    }

    for res in results:
        nm = res["name"]
        x_pt = res["d50"]
        y_pt = res["EC"]
        c_pt = colors[nm]
        ax.scatter([x_pt], [y_pt], color=c_pt, s=120, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"{nm} (d50={int(x_pt):,}m)\nEC: {y_pt:,.1f} ha",
            (x_pt, y_pt),
            xytext=offsets[nm],
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=c_pt,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_pt, alpha=0.85),
            zorder=6
        )

    ax.set_xscale("log")
    ax.set_title("종별 분산거리($d_{50}$) 변화에 따른 서울 녹지망 등가 연결 면적(EC) 민감도", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("분산거리 중앙값 $d_{50}$ (m, 로그 눈금)", fontsize=11, fontweight="bold")
    ax.set_ylabel("등가 연결 면적 EC (ha)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_ylim(5000, 18000)

    # 녹지 총면적 및 단절 시 기준선 표시
    ax.axhline(total_area_ha, color="#757575", linestyle=":", linewidth=1.2, label=f"녹지 총면적 ({total_area_ha:,.1f} ha)")
    isolated_ec = float(np.sqrt(np.sum(a ** 2)))
    ax.axhline(isolated_ec, color="#9e9e9e", linestyle="--", linewidth=1.2, label=f"완전 단절 시 EC ({isolated_ec:,.1f} ha)")

    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_curve_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  - {output_curve_png} 저장 완료.")

    total_elapsed = time.time() - total_start_time
    print(f"\n=== [프롬프트 5] 전체 작업 완료 (총 소요 시간: {total_elapsed:.1f}초) ===")


if __name__ == "__main__":
    main()
