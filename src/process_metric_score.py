import os
import re
from openpyxl import Workbook

def extract_metrics_from_txt(txt_file):
    try:
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            if len(lines) < 2:  
                return None
            last_line = lines[-1]
            second_last_line = lines[-2]

            metrics_last = re.findall(r"SROCC=([-+]?\d*\.\d+|\d+).*KROCC=([-+]?\d*\.\d+|\d+).*PLCC=([-+]?\d*\.\d+|\d+).*RMSE=([-+]?\d*\.\d+|\d+)", last_line)
            return metrics_last[0]

    except Exception as e:
        print(f"Error processing {txt_file}: {e}")
        return None

def main(source_directory, output_file):

    wb = Workbook()
    ws = wb.active


    txt_files = [file for file in os.listdir(source_directory) if file.endswith('.txt')]

    for i, txt_file in enumerate(txt_files):
        txt_path = os.path.join(source_directory, txt_file)
        metrics = extract_metrics_from_txt(txt_path)
        if metrics:
            row = [f"File {i + 1}"]
            row.extend(metrics)
            ws.append(row)
        else:
            print(f"Skipping {txt_path} due to error or missing data")


    wb.save(output_file)

if __name__ == "__main__":
    source_directory = "/workspace/CBAND/logs/LIVE_Banding_vgg16_stage2_ggd2_txt/"  
    output_file = "/workspace/CBAND/logs/LIVE_Banding_vgg16_stage2_ggd2_metric.xlsx"  
    main(source_directory, output_file)
