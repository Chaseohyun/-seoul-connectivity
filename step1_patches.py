# step1_patches.py: NDVI 계산 및 녹지 패치 생성 스크립트
# 규칙 준수: 한국어 주석, 기준값 하드코딩 배제, 지정된 8개 라이브러리만 활용

import os
import sys
import time
import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import shape
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    start_time = time.time()
    print("=== [프롬프트 1] NDVI 계산 및 녹지 패치 생성 시작 ===")

    # 1. 입력 파일 경로
    input_tif = "seoul_s2_20260516.tif"
    boundary_gpkg = "seoul_boundary.gpkg"
    output_ndvi_tif = "seoul_ndvi.tif"
    output_patches_gpkg = "green_patches.gpkg"
    output_summary = "step1_summary.txt"
    output_map = "step1_map.png"

    # 2. Sentinel-2 영상 읽기
    print("\n1. 위성영상 읽기 및 메타데이터 확인...")
    with rasterio.open(input_tif) as src:
        b1 = src.read(1)  # 밴드 1: B04 (적색, Red)
        b2 = src.read(2)  # 밴드 2: B08 (근적외, NIR)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)
    pixel_area_m2 = pixel_width * pixel_height  # 화소 면적 (m²)

    # 3. 유효 화소 마스크 및 NDVI 계산
    # 규칙: 두 밴드가 모두 0보다 큰 화소만 계산, 나머지는 NoData(-9999)
    print("2. NDVI 계산 및 기술통계 산출...")
    valid_mask = (b1 > 0) & (b2 > 0)
    valid_pixel_count = int(np.count_nonzero(valid_mask))
    valid_area_km2 = (valid_pixel_count * pixel_area_m2) / 1_000_000.0

    b1_valid = b1[valid_mask].astype(np.float32)
    b2_valid = b2[valid_mask].astype(np.float32)

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi_valid = (b2_valid - b1_valid) / (b2_valid + b1_valid)

    ndvi_array = np.full(b1.shape, -9999.0, dtype=np.float32)
    ndvi_array[valid_mask] = ndvi_valid

    # NDVI 통계치
    ndvi_mean = float(np.mean(ndvi_valid))
    ndvi_median = float(np.median(ndvi_valid))
    ndvi_min = float(np.min(ndvi_valid))
    ndvi_max = float(np.max(ndvi_valid))

    print(f"  - 자료 화소 수: {valid_pixel_count:,}개 ({valid_area_km2:.2f} km²)")
    print(f"  - NDVI 통계: 평균 {ndvi_mean:.3f}, 중앙값 {ndvi_median:.3f}, 최소 {ndvi_min:.3f}, 최대 {ndvi_max:.3f}")

    # seoul_ndvi.tif 파일 저장
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999.0)
    with rasterio.open(output_ndvi_tif, "w", **profile) as dst:
        dst.write(ndvi_array, 1)
    print(f"  - {output_ndvi_tif} 저장 완료.")

    # 4. 녹지 화소 정의 및 패치 추출 (8-연결)
    print("\n3. 녹지 패치 추출 및 벡터화...")
    # 녹지 = NDVI >= 0.6인 화소
    green_mask = valid_mask & (ndvi_array >= 0.6)
    green_pixel_count = int(np.count_nonzero(green_mask))
    green_area_ha = (green_pixel_count * pixel_area_m2) / 10000.0
    green_ratio_pct = (green_pixel_count / valid_pixel_count) * 100.0

    print(f"  - 녹지 화소 수: {green_pixel_count:,}개 = {green_area_ha:.1f} ha (서울 자료 면적의 {green_ratio_pct:.1f} %)")

    # 8-연결 라벨링
    structure_8 = np.ones((3, 3), dtype=bool)
    labeled_array, num_patches = ndimage.label(green_mask, structure=structure_8)
    print(f"  - 8-연결 기준 녹지 패치 수: {num_patches:,}개")

    # 벡터 폴리곤 변환 (connectivity=8)
    shapes_generator = features.shapes(
        labeled_array.astype(np.int32),
        mask=labeled_array > 0,
        connectivity=8,
        transform=transform
    )

    geometries = []
    for geom_dict, label_val in shapes_generator:
        geom = shape(geom_dict)
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
        geometries.append(geom)

    # 면적 및 중심점 좌표 계산
    areas_ha = [geom.area / 10000.0 for geom in geometries]
    centroid_y = [geom.centroid.y for geom in geometries]
    centroid_x = [geom.centroid.x for geom in geometries]

    df_patches = pd.DataFrame({
        "geometry": geometries,
        "ha": areas_ha,
        "c_y": centroid_y,
        "c_x": centroid_x
    })

    # 정렬 규칙: 면적 내림차순, 동일 면적 시 중심점 y 내림차순, 그다음 x 오름차순
    df_patches = df_patches.sort_values(
        by=["ha", "c_y", "c_x"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    df_patches["pid"] = np.arange(1, len(df_patches) + 1, dtype=int)

    # GeoDataFrame 구성 (필드: pid, ha, geometry)
    gdf_patches = gpd.GeoDataFrame(
        df_patches[["pid", "ha", "geometry"]],
        geometry="geometry",
        crs=crs
    )

    # GeoPackage 저장 (레이어 이름 patches)
    gdf_patches.to_file(output_patches_gpkg, layer="patches", driver="GPKG", engine="pyogrio")
    print(f"  - {output_patches_gpkg} (레이어: patches) 저장 완료.")

    # 5. 통계 분석 및 요약문 작성
    print("\n4. 패치 통계 집계 및 요약문 작성...")
    # 단일 화소 패치 면적 (0.01 ha)
    single_pixel_ha = pixel_area_m2 / 10000.0
    single_pixel_patches = int(np.count_nonzero(np.isclose(gdf_patches["ha"], single_pixel_ha)))

    under_1ha = gdf_patches[gdf_patches["ha"] < 1.0]
    under_1ha_count = len(under_1ha)
    under_1ha_total_ha = float(under_1ha["ha"].sum())

    over_1ha = gdf_patches[gdf_patches["ha"] >= 1.0]
    over_1ha_count = len(over_1ha)
    over_1ha_total_ha = float(over_1ha["ha"].sum())

    top1_ha = float(gdf_patches.iloc[0]["ha"])
    top2_ha = float(gdf_patches.iloc[1]["ha"])
    top3_ha = float(gdf_patches.iloc[2]["ha"])

    summary_content = f"""[프롬프트 1] 녹지 패치 및 연결성 분석 결과 요약
--------------------------------------------------------------
- 자료 화소 수: {valid_pixel_count:,}개 ({valid_area_km2:.2f} km²)
- NDVI 평균: {ndvi_mean:.3f}, 중앙값: {ndvi_median:.3f}, 최소: {ndvi_min:.3f}, 최대: {ndvi_max:.3f}
- 녹지 화소: {green_pixel_count:,}개 = {green_area_ha:.1f} ha (서울의 {green_ratio_pct:.1f} %)
- 패치 수: {len(gdf_patches):,}개
- 가장 큰 패치: {top1_ha:.1f} ha, 둘째: {top2_ha:.1f} ha, 셋째: {top3_ha:.1f} ha
- 화소 1개 패치: {single_pixel_patches:,}개
- 1 ha 미만 패치: {under_1ha_count:,}개 (면적: {under_1ha_total_ha:.1f} ha)
- 1 ha 이상 패치: {over_1ha_count:,}개 (면적: {over_1ha_total_ha:.1f} ha)
--------------------------------------------------------------
"""
    with open(output_summary, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"  - {output_summary} 저장 완료.")

    # 6. 지도 시각화 (step1_map.png)
    print("\n5. 시각화 지도 생성 (step1_map.png)...")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    # 서울 경계 벡터 레이어 읽기
    gu_boundary = gpd.read_file(boundary_gpkg, layer="gu", engine="pyogrio")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

    # 지리적 범위 계산
    extent = [
        transform.c,
        transform.c + transform.a * width,
        transform.f + transform.e * height,
        transform.f
    ]

    # 왼쪽 패널: NDVI 지도
    ndvi_display = np.where(valid_mask, ndvi_array, np.nan)
    im0 = axes[0].imshow(ndvi_display, cmap="YlGn", extent=extent, vmin=-0.2, vmax=0.9, origin="upper")
    gu_boundary.boundary.plot(ax=axes[0], color="#555555", linewidth=0.5, linestyle="--")
    axes[0].set_title("서울 식생지수 (NDVI)", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_xlabel("X 좌표 (m, EPSG:5186)")
    axes[0].set_ylabel("Y 좌표 (m, EPSG:5186)")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("NDVI")

    # 오른쪽 패널: 녹지 패치 지도
    gu_boundary.plot(ax=axes[1], facecolor="#f3f4f6", edgecolor="#666666", linewidth=0.6)
    gdf_patches.plot(
        ax=axes[1],
        column="ha",
        cmap="YlGn",
        norm=mcolors.LogNorm(vmin=0.01, vmax=5000),
        legend=True,
        legend_kwds={"label": "패치 면적 (ha, 로그 눈금)", "orientation": "vertical", "fraction": 0.046, "pad": 0.04}
    )
    axes[1].set_title(f"녹지 패치 분포 (총 {len(gdf_patches):,}개)", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xlabel("X 좌표 (m, EPSG:5186)")
    axes[1].set_ylabel("Y 좌표 (m, EPSG:5186)")
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])

    plt.tight_layout()
    plt.savefig(output_map, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  - {output_map} 저장 완료.")

    elapsed = time.time() - start_time
    print(f"\n=== [프롬프트 1] 전체 실행 완료 (소요 시간: {elapsed:.1f}초) ===")


if __name__ == "__main__":
    main()
