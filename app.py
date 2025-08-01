from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # 用于存储上传文件的目录
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 1. 启动时加载分类器资产 ---
imputer = None
model = None
feature_names = []

# 定义特征描述的字典
# 这个字典应该在Flask应用启动时就加载好，并且是全局可访问的
feature_descriptions = {
    'Age': '患者的年龄（单位：年）',
    'Gender': '患者的性别（0表示女性，1表示男性）。例如：0 或 1',
    'Ethnicity': '患者的种族（0：白种人 1：非裔美国人 2：亚裔 3：其他）',
    'Education': '患者的教育程度（0：无 、1：高中 、2：本科 、3：更高） ',
    'BMI': '患者的体质指数（BMI值）,一般范围为15-40',
    'Smoking': '患者吸烟状况，0代表不抽烟，1代表抽烟',
    'AlcoholConsumption': '每周酒精摄入量，范围从0-20',
    'PhysicalActivity':'每周体力活动量（小时）',
    'DietQuality':'饮食质量评分，范围0-10',
    'SleepQuality':'睡眠质量评分，范围4-10',
    'FamilyHistoryAlzheimers':'阿尔茨海默病家族史，0 表示无，1 表示有',
    'CardiovascularDisease':'心血管疾病的存在，0 表示无，1 表示有',
    'Diabetes':'糖尿病的存在，0 表示无，1 表示有',
    'Depression':'抑郁症的存在，0 表示无，1 表示有',
    'HeadInjury':'头部损伤史，0 表示无，1 表示有',
    'Hypertension':'高血压的存在，0 表示无，1 表示有',
    'SystolicBP':'收缩压，范围：90 至 180 毫米汞柱',
    'DiastolicBP':'舒张压，范围：60 至 120 毫米汞柱',
    'CholesterolTotal':'总胆固醇水平，范围：150 至 300 毫克/分升',
    'CholesterolLDL':'低密度脂蛋白胆固醇水平，范围：50 至 200 毫克/分升',
    'CholesterolHDL':'高密度脂蛋白胆固醇水平，范围：20 至 100 毫克/分升',
    'CholesterolTriglycerides':'甘油三酯水平，范围：50 至 400 毫克/分升',
    'MMSE': '简易精神状态检查量表评分，范围：0 至 30。分数较低表示认知障碍',
    'FunctionalAssessment':'功能评估评分，范围为 0 至 10 分。分数越低，大脑功能受损程度越严重',
    'MemoryComplaints':'存在记忆障碍，0 表示无，1 表示有',
    'BehavioralProblems':'存在行为问题，0 表示无，1 表示有',
    'ADL':'日常生活活动能力评分，范围为 0 至 10 分。分数越低，受损程度越严重。 ',
    'Confusion':'存在意识混乱，0 表示无，1 表示有',
    'Disorientation':'存在定向障碍，0 表示无，1 表示有',
    'PersonalityChanges':'存在人格改变，0 表示无，1 表示有',
    'DifficultyCompletingTasks':'存在完成任务困难，0 表示无，1 表示有',
    'Forgetfulness':'存在健忘症状，0 表示无，1 表示有',
    # 根据你的实际特征名和需要添加更多描述
}

try:
    imputer = joblib.load('model/imputer.joblib')
    model = joblib.load('model/AD_Prediction.joblib')
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        feature_names = imputer.feature_names_in_
    print("✅ 分类模型和数据填充器加载成功！")
    print(f"实际加载的特征名: {feature_names}") # 调试信息
except FileNotFoundError as e:
    print(f"❌ 错误: 必需的模型文件未找到 -> {e}")
    print("请确保 'model/' 目录下有 'AD_Prediction.joblib' 和 'imputer.joblib' 文件。")
    # 如果模型加载失败，使用一个默认的特征名列表，确保应用能启动
    feature_names = list(feature_descriptions.keys()) # 使用字典的键作为默认特征名
except AttributeError:
    print("❌ 错误: 无法从模型或填充器中确定特征名称。")
    # 如果无法确定特征名，也使用字典的键作为默认特征名
    feature_names = list(feature_descriptions.keys())


# 预测函数
def predict_diagnosis(data_df):
    global imputer, model, feature_names
    if imputer is None or model is None:
        print("错误：模型或数据填充器未加载，无法进行预测。")
        return pd.DataFrame()  # 返回空DataFrame或错误信息

    # 确保输入数据的列与模型训练时的特征列一致
    processed_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in data_df.columns:
            # 尝试转换为数值，非数值的将变为NaN
            processed_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        else:
            processed_df[col] = np.nan  # 缺失的列用NaN填充

    # 确保processed_df的列顺序和imputer期望的一致
    # 这一步非常重要，否则imputer和model可能因为列顺序不一致而出错
    processed_df = processed_df[feature_names]

    imputed_data = imputer.transform(processed_df)
    probabilities = model.predict_proba(imputed_data)
    diagnosis_probability = probabilities[:, 1]  # 确诊的概率在第二列 (索引为 1)

    results_df = data_df.copy()  # 保留原始输入数据
    results_df['diagnosis_probability'] = np.round(diagnosis_probability, 4)
    return results_df


# 核心路由函数，只定义一次
@app.route('/', methods=['GET', 'POST'])
def index():
    # 声明 feature_names 和 feature_descriptions 为全局变量，以便在函数内访问
    global feature_names, feature_descriptions
    results_df_html = None
    error_message = None

    if request.method == 'POST':
        # --- 手动输入数据处理 ---
        if 'submit_manual' in request.form:
            manual_data = {}
            for feature in feature_names:
                value = request.form.get(f'feature_{feature}')
                if value:
                    # 将 'NA' 或 'na' 转换为 np.nan
                    manual_data[feature] = [value if value.strip().lower() != 'na' else np.nan]
                else:
                    manual_data[feature] = [np.nan]  # 如果用户没填，也视为缺失

            try:
                input_df = pd.DataFrame(manual_data)
                # 确保所有特征都是数值类型，非数值的转换为 NaN
                for col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

                results_df = predict_diagnosis(input_df)
                if not results_df.empty:
                    # 使用 to_html() 方法将 DataFrame 转换为 HTML 表格
                    results_df_html = results_df.to_html(classes='table table-striped', index=False)
                else:
                    error_message = "预测失败，请检查模型加载是否成功或输入数据格式。也可能是所有输入都是缺失值导致无法预测。"
            except Exception as e:
                error_message = f"处理手动输入数据时发生错误: {e}"

        # --- 文件上传数据处理 ---
        elif 'submit_file' in request.form:
            if 'file' not in request.files:
                error_message = '没有文件部分被选中。'
            else:
                file = request.files['file']
                if file.filename == '':
                    error_message = '没有选择任何文件。'
                elif file:
                    try:
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                        file.save(filepath)

                        # 根据文件扩展名读取数据，处理大小写不敏感
                        if file.filename.lower().endswith('.csv'):
                            input_df = pd.read_csv(filepath)
                        elif file.filename.lower().endswith(('.xls', '.xlsx')):
                            input_df = pd.read_excel(filepath)
                        else:
                            error_message = '不支持的文件类型。请上传 CSV 或 Excel 文件。'
                            os.remove(filepath)  # 删除不支持的文件
                            # 提前返回，避免后续处理
                            return render_template('index.html',
                                                   feature_names=feature_names,
                                                   feature_descriptions=feature_descriptions,
                                                   results_html=results_df_html,
                                                   error=error_message)

                        os.remove(filepath)  # 预测完成后删除上传的临时文件

                        results_df = predict_diagnosis(input_df)
                        if not results_df.empty:
                            results_df_html = results_df.to_html(classes='table table-striped', index=False)
                        else:
                            error_message = "预测失败，请检查模型加载是否成功或文件数据格式。请确保文件包含正确的特征列。"
                    except Exception as e:
                        error_message = f"处理文件时发生错误: {e}"

    # 无论 GET 请求还是 POST 请求处理完后，都渲染模板并传递所有必要数据
    return render_template('index.html',
                           feature_names=feature_names,
                           feature_descriptions=feature_descriptions, # <-- 确保这里正确传递了字典
                           results_html=results_df_html,
                           error=error_message)

if __name__ == '__main__':
    app.run(debug=True)  # debug=True 可以在代码修改后自动重启服务器
