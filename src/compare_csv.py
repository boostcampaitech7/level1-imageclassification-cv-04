import pandas as pd
import numpy as np

def compare_csv_files(file1_path: str, file2_path: str, id_column: str = 'ID'):
    """
    두 CSV 파일을 행 기준으로 비교하는 함수
    
    :param file1_path: 첫 번째 CSV 파일 경로
    :param file2_path: 두 번째 CSV 파일 경로
    :param id_column: 행을 식별하는 데 사용할 열 이름 (기본값: 'ID')
    :return: None
    """
    # CSV 파일 읽기
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # ID 열이 존재하는지 확인
    if id_column not in df1.columns or id_column not in df2.columns:
        raise ValueError(f"'{id_column}' 열이 두 CSV 파일 모두에 존재하지 않습니다.")
    
    # ID 열을 인덱스로 설정
    df1.set_index(id_column, inplace=True)
    df2.set_index(id_column, inplace=True)
    
    # 공통 인덱스만 선택
    common_indices = df1.index.intersection(df2.index)
    df1_common = df1.loc[common_indices]
    df2_common = df2.loc[common_indices]
    
    # 열 비교
    columns_diff = set(df1.columns).symmetric_difference(set(df2.columns))
    if columns_diff:
        print(f"두 파일의 열이 다릅니다. 차이: {columns_diff}")
        common_columns = list(set(df1.columns).intersection(set(df2.columns)))
        df1_common = df1_common[common_columns]
        df2_common = df2_common[common_columns]
    
    # 값 비교
    differences = (df1_common != df2_common) & ~(df1_common.isna() & df2_common.isna())
    diff_count = differences.sum().sum()
    
    print(f"공통 행에서 {diff_count}개의 값이 다릅니다.")
    
    # if diff_count > 0:
    #     print("\n값이 다른 셀의 상세 정보:")
    #     for col in differences.columns:
    #         diff_indices = differences.index[differences[col]]
    #         for idx in diff_indices:
    #             print(f"행 {idx}, 열 '{col}': {df1_common.loc[idx, col]} vs {df2_common.loc[idx, col]}")

# 함수 사용 예시
compare_csv_files("/data/ephemeral/home/deamin/level1-imageclassification-cv-04/output_soft_voting.csv", 
                "/data/ephemeral/home/deamin/level1-imageclassification-cv-04/output_wrong_sVoting.csv") #/data/ephemeral/home/deamin/level1-imageclassification-cv-04/pj1_lr_00001_output.csv