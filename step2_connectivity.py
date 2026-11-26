# step2_connectivity.py: 패치 사이 연결성(PC, EC, dPC) 산정 스크립트
# 규칙 준수: 한국어 주석, 기준값 하드코딩 배제, 8개 기본 패키지만 사용

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import multiprocessing as mp

# 워커 프로세스용 전역 변수
g_graph = None
g_a = None
g_decay = None


def init_worker(shared_graph, shared_a, shared_decay):
    """각 워커 프로세스 초기화 시 그래프 및 속성 공유"""
    global g_graph, g_a, g_decay
    g_graph = shared_graph
    g_a = shared_a
    g_decay = shared_decay


def process_batch(args):
    """지정된 범위의 노드들에 대해 다익스트라 최단 경로 및 p* flux 계산"""
    start_idx, end_idx = args
    indices = np.arange(start_idx, end_idx)

    # 1. 최단 경로 링크 거리 합 계산 (Dijkstra)
    D = dijkstra(g_graph, directed=False, indices=indices)

    # 2. 경로 연결 확률 p* = exp(-ln(2)/250 * D)
    # D가 inf(도달 불가)인 경우 np.exp(-inf) = 0.0 처리
    P = np.exp(g_decay * D)

    # 3. 각 노드별 flux_ha 계산: Σ_{j≠i} a_j * p*_ij
    # P @ g_a = Σ_j a_j * p*_ij (자기 자신 포함: a_i * p*_ii = a_i)
    total_flux_and_self = P @ g_a
    flux_batch = total_flux_and_self - g_a[indices]

    # 4. 상위 2000개 노드 간 p* 행렬 추출 (프롬프트 3용)
    pstar_sub = None
    if start_idx < 2000:
        actual_end = min(end_idx, 2000)
        pstar_sub = (start_idx, actual_end, P[: actual_end - start_idx, :2000].astype(np.float32))

    return start_idx, end_idx, flux_batch, pstar_sub


def main():
    total_start_time = time.time()
    print("=== [프롬프트 2] 패치 사이 연결성(PC · EC · dPC) 산정 시작 ===")

    # 1. 입력 및 출력 파일 경로
    input_patches_gpkg = "green_patches.gpkg"
    output_links_csv = "patch_links.csv"
    output_conn_gpkg = "patch_connectivity.gpkg"
    output_pstar_npy = "pstar_top2000.npy"
    output_summary = "step2_summary.txt"

    # 2. 패치 데이터 읽기
    print("\n1. 패치 데이터(green_patches.gpkg) 읽기...")
    gdf = gpd.read_file(input_patches_gpkg, layer="patches", engine="pyogrio")
    gdf = gdf.sort_values("pid").reset_index(drop=True)
    N = len(gdf)
    geometries = gdf.geometry.values
    a = gdf["ha"].values.astype(np.float64)
    total_green_ha = float(np.sum(a))
    print(f"  - 총 패치 수: {N:,}개, 녹지 면적 합: {total_green_ha:.1f} ha")

    # 3. 직접 링크 탐색 (500m 이내) 및 최단 거리 계산
    print("\n2. STRtree 공간 인덱스를 활용한 직접 링크(500m 이하) 탐색...")
    tree_start = time.time()
    tree = shapely.STRtree(geometries)
    query_res = tree.query(geometries, predicate="dwithin", distance=500.0)

    # 중복 및 자기 자신 쌍 제거 (i < j 만 유지)
    mask_ij = query_res[0] < query_res[1]
    idx_a = query_res[0][mask_ij]
    idx_b = query_res[1][mask_ij]

    # 정확한 최단 거리(m) 계산
    edge_dists = shapely.distance(geometries[idx_a], geometries[idx_b])
    valid_mask = edge_dists <= 500.0
    idx_a = idx_a[valid_mask]
    idx_b = idx_b[valid_mask]
    edge_dists = edge_dists[valid_mask]

    num_links = len(idx_a)
    min_dist = float(np.min(edge_dists))
    median_dist = float(np.median(edge_dists))
    max_dist = float(np.max(edge_dists))

    print(f"  - 탐색 완료 소요 시간: {time.time() - tree_start:.1f}초")
    print(f"  - 직접 링크 수: {num_links:,}쌍 (최소: {min_dist:.1f} m, 중앙값: {median_dist:.1f} m, 최대: {max_dist:.1f} m)")

    # patch_links.csv 저장
    print("\n3. 직접 링크 목록 저장 (patch_links.csv)...")
    df_links = pd.DataFrame({
        "pid_a": idx_a + 1,
        "pid_b": idx_b + 1,
        "dist_m": edge_dists
    })
    df_links.to_csv(output_links_csv, index=False)
    print(f"  - {output_links_csv} 저장 완료.")

    # 4. 희소 인접 행렬 구성 (CSR Matrix)
    print("\n4. 최단 경로 계산용 희소 그래프 행렬 생성...")
    row_indices = np.concatenate([idx_a, idx_b])
    col_indices = np.concatenate([idx_b, idx_a])
    data_weights = np.concatenate([edge_dists, edge_dists])
    graph_csr = csr_matrix((data_weights, (row_indices, col_indices)), shape=(N, N))

    # 감쇄 계수: p = 0.5^(d/250) = exp(-ln(2)/250 * d)
    decay = -np.log(2.0) / 250.0

    # 5. 다중 프로세스 배치 다익스트라 실행 (500개씩 분할)
    batch_size = 500
    batches = [(i, min(i + batch_size, N)) for i in range(0, N, batch_size)]
    num_cpus = max(1, min(mp.cpu_count() - 2, 12))  # 적정 워커 수 지정
    print(f"\n5. 다중 프로세스(워커 {num_cpus}개) 배치 다익스트라 연산 시작 (총 {len(batches)}개 배치)...")

    dijkstra_start = time.time()
    flux_ha = np.zeros(N, dtype=np.float64)
    pstar_top2000 = np.zeros((2000, 2000), dtype=np.float32)

    with mp.Pool(processes=num_cpus, initializer=init_worker, initargs=(graph_csr, a, decay)) as pool:
        completed_batches = 0
        for s_idx, e_idx, f_batch, p_sub in pool.imap_unordered(process_batch, batches):
            flux_ha[s_idx:e_idx] = f_batch
            if p_sub is not None:
                ps_s, ps_e, ps_mat = p_sub
                pstar_top2000[ps_s:ps_e, :] = ps_mat
            completed_batches += 1
            if completed_batches % 5 == 0 or completed_batches == len(batches):
                print(f"  - 진행률: {completed_batches}/{len(batches)} 배치 완료 ({time.time() - dijkstra_start:.1f}초 경과)")

    print(f"  - 다익스트라 연산 완료 소요 시간: {time.time() - dijkstra_start:.1f}초")

    # pstar_top2000.npy 저장
    print("\n6. pstar_top2000.npy 저장...")
    np.save(output_pstar_npy, pstar_top2000)
    print(f"  - {output_pstar_npy} 저장 완료 (형상: {pstar_top2000.shape}).")

    # 6. 연결성 지표(Q, PC, EC, dPC) 산정
    print("\n7. 종합 연결성 지표 계산...")
    # Q = Σ_i a_i * (a_i + flux_ha_i)
    Q = float(np.sum(a * (a + flux_ha)))

    # 서울 경계 내 화소 면적 A_L = 60,668.09 ha
    A_L = 60668.09
    PC = Q / (A_L ** 2)
    EC = float(np.sqrt(Q))
    EC_pct = (EC / total_green_ha) * 100.0

    # 단절 상태 EC = √(Σ a_i²)
    EC_isolated = float(np.sqrt(np.sum(a ** 2)))

    # 패치별 dPC 계산
    dPC_intra = 100.0 * (a ** 2) / Q
    dPC_flux = 100.0 * (2.0 * a * flux_ha) / Q
    dPC_if = dPC_intra + dPC_flux

    # 결과 데이터프레임 구성 및 순위 산정
    gdf["flux_ha"] = flux_ha
    gdf["dPC_intra"] = dPC_intra
    gdf["dPC_flux"] = dPC_flux
    gdf["dPC_if"] = dPC_if
    gdf["rank_if"] = gdf["dPC_if"].rank(ascending=False, method="min").astype(int)

    # patch_connectivity.gpkg 저장
    print("\n8. 연결성 결과 벡터 저장 (patch_connectivity.gpkg)...")
    out_cols = ["pid", "ha", "flux_ha", "dPC_intra", "dPC_flux", "dPC_if", "rank_if", "geometry"]
    gdf_out = gdf[out_cols].copy()
    gdf_out.to_file(output_conn_gpkg, layer="patches", driver="GPKG", engine="pyogrio")
    print(f"  - {output_conn_gpkg} (레이어: patches) 저장 완료.")

    # 주요 패치 쌍 p* 확인 (0-indexed 변환: pid k -> k-1)
    p_1_2 = float(pstar_top2000[0, 1])
    p_1_3 = float(pstar_top2000[0, 2])
    p_1_8 = float(pstar_top2000[0, 7])
    p_1_10 = float(pstar_top2000[0, 9])
    p_2_6 = float(pstar_top2000[1, 5])

    # 상위 dPC_if 패치 정보
    top_pids = [1, 2, 3, 4, 8]
    top_info_lines = []
    for t_pid in top_pids:
        row = gdf[gdf["pid"] == t_pid].iloc[0]
        top_info_lines.append(
            f"  - pid {t_pid}: dPC_if {row['dPC_if']:.2f} % (intra {row['dPC_intra']:.2f} + flux {row['dPC_flux']:.2f}), 순위: {int(row['rank_if'])}"
        )

    total_elapsed = time.time() - total_start_time

    # 요약 텍스트 작성
    summary_content = f"""[프롬프트 2] 패치 사이 연결성(PC · EC · dPC) 계산 결과 요약
--------------------------------------------------------------
- 총 실행 소요 시간: {total_elapsed:.1f}초
- 직접 링크: {num_links:,}쌍 (최소 거리: {min_dist:.1f} m, 중앙값: {median_dist:.1f} m, 최대: {max_dist:.1f} m)
- Q: {Q:,.0f} ha²
- PC: {PC:.5f}
- EC: {EC:,.1f} ha (녹지 면적 합 {total_green_ha:,.1f} ha의 {EC_pct:.1f} %)
- 단절 시 EC: {EC_isolated:,.1f} ha
- 주요 패치 간 최대 경로 확률(p*):
    pid 1 – pid 2: {p_1_2:.3f}
    pid 1 – pid 3: {p_1_3:.3f}
    pid 1 – pid 8: {p_1_8:.3f}
    pid 1 – pid 10: {p_1_10:.3f}
    pid 2 – pid 6: {p_2_6:.3f}
- 상위 패치 dPC_if 기여도:
{chr(10).join(top_info_lines)}
--------------------------------------------------------------
"""
    with open(output_summary, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"\n  - {output_summary} 저장 완료.")

    print(f"\n=== [프롬프트 2] 전체 작업 완료 (총 소요 시간: {total_elapsed:.1f}초) ===")


if __name__ == "__main__":
    main()
