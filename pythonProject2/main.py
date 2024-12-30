# 导入必要的库
import librosa  # 使用librosa库来提取音频特征
import librosa.feature
import os
import numpy as np
import soundfile as sf
from sklearn.mixture import GaussianMixture  # GMM用于模型训练
from hmmlearn.hmm import GaussianHMM  # HMM用于序列建模
import pickle  # 用于模型保存与加载
from scipy.signal import resample  # 用于音频重采样
import tkinter as tk  # GUI库
from tkinter import filedialog, messagebox  # 用于文件选择和消息框
import sounddevice as sd  # 用于录制音频
import wave  # 用于音频文件读写

def process_and_extract_features(audio_path, num_coeff=13, nfft=2048, max_len=100, target_samplerate=16000, max_duration=3):
    """
    对音频进行预处理：重采样、归一化、截断或填充，并提取MFCC特征，增加ΔMFCC和ΔΔMFCC进行增强。
    """
    try:
        # 读取音频文件
        audio_data, samplerate = sf.read(audio_path)
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")

        # 进行重采样、归一化和填充/截断
        if samplerate != target_samplerate:
            num_samples = int(target_samplerate * len(audio_data) / samplerate)
            audio_data = resample(audio_data, num_samples)  # 重采样
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)  # 归一化

        # 截断或填充音频到指定时长
        max_samples = target_samplerate * max_duration
        if len(audio_data) > max_samples:
            audio_data = audio_data[:max_samples]  # 截断音频
        elif len(audio_data) < max_samples:
            padding = max_samples - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')  # 填充音频

        # 提取MFCC特征，ΔMFCC和ΔΔMFCC
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=target_samplerate, n_mfcc=num_coeff, n_fft=nfft)
        delta_features = librosa.feature.delta(mfcc_features)  # 计算ΔMFCC
        delta_delta_features = librosa.feature.delta(delta_features)  # 计算ΔΔMFCC

        # 合并MFCC、ΔMFCC和ΔΔMFCC特征
        combined_features = np.hstack((mfcc_features, delta_features, delta_delta_features))
        n_samples, n_features = combined_features.shape

        # 截断或填充特征数据到指定长度
        if n_samples > max_len:
            combined_features = combined_features[:max_len, :]
        elif n_samples < max_len:
            pad_width = max_len - n_samples
            combined_features = np.pad(combined_features, ((0, pad_width), (0, 0)), mode='constant')

        # 对特征进行正则化处理
        mean = np.mean(combined_features, axis=0)
        std = np.std(combined_features, axis=0)
        std = np.clip(std, 1e-8, None)  # 避免标准差过小带来的问题
        normalized_features = (combined_features - mean) / std

        return normalized_features
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None


def train_gmm_hmm(audio_files, label, num_gmm_components=5, hmm_states=5, max_iter=3000, tol=1e-6):
    """
    使用GMM-HMM训练一个字母的语音模型。
    """
    all_features = []

    # 提取每个音频文件的特征
    for audio_file in audio_files:
        features = process_and_extract_features(audio_file)
        if features is not None:
            all_features.append(features)

    if len(all_features) == 0:
        print(f"No valid features for label '{label}', skipping training.")
        return None

    all_features = np.concatenate(all_features, axis=0)  # 合并所有音频文件的特征

    try:
        # 使用GMM对特征进行训练
        gmm = GaussianMixture(n_components=num_gmm_components, covariance_type='diag', max_iter=max_iter, tol=tol)
        gmm.fit(all_features)

        # 初始化HMM模型，采用高斯分布（GaussianHMM），不进行参数初始化
        model = GaussianHMM(n_components=hmm_states, covariance_type='diag', n_iter=max_iter, tol=tol,
                            init_params='')
        model.startprob_ = np.full(hmm_states, 1.0 / hmm_states)  # 设置初始状态概率
        model.transmat_ = np.full((hmm_states, hmm_states), 1.0 / hmm_states)  # 设置状态转移概率
        model.means_ = gmm.means_  # 设置高斯分布的均值
        model.covars_ = gmm.covariances_  # 设置高斯分布的协方差

        # 使用HMM对特征进行训练
        model.fit(all_features)
        save_model(label, model)  # 保存训练好的模型
        return model
    except Exception as e:
        print(f"Error training model for label '{label}': {e}")
        return None


def save_model(label, model):
    """
    保存训练好的模型到指定路径。
    """
    model_path = os.path.join('models', f'{label}_gmm_hmm.pkl')  # 模型文件的保存路径
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)  # 使用pickle保存模型


def load_model(label):
    """
    从指定路径加载训练好的模型。
    """
    model_path = os.path.join('models', f'{label}_gmm_hmm.pkl')
    if os.path.exists(model_path):  # 如果模型文件存在
        with open(model_path, 'rb') as f:
            return pickle.load(f)  # 加载并返回模型
    else:
        print(f"Warning: Model for label '{label}' not found.")  # 如果没有找到模型，打印警告
        return None


def predict_with_gmm_hmm(models, test_audio_path):
    """
    使用GMM-HMM模型预测音频中的字符。
    """
    features = process_and_extract_features(test_audio_path)  # 提取音频特征
    if features is None:
        print("No valid features extracted from test audio.")  # 如果特征提取失败，返回None
        return None

    best_label = None
    best_score = float('-inf')  # 初始化最佳分数为负无穷

    # 遍历每个字母的模型
    for label, model in models.items():
        try:
            log_likelihood, _ = model.decode(features)  # 使用模型解码特征，得到对数似然值
            if log_likelihood > best_score:  # 如果当前模型的对数似然值更高，更新最佳标签和分数
                best_score = log_likelihood
                best_label = label
        except Exception as e:
            print(f"Error decoding with model '{label}': {e}")  # 解码失败时打印错误信息

    return best_label  # 返回最匹配的字母标签


def train_models():
    """
    训练所有字母的GMM-HMM模型。
    """
    if not os.path.exists('models'):  # 如果模型保存目录不存在，则创建
        os.makedirs('models')

    audio_files_by_label = {label: [] for label in 'abcdefghijklmnopqrstuvwxyz'}  # 初始化字母对应的音频文件列表
    # 遍历字母a-z
    for label in 'abcdefghijklmnopqrstuvwxyz':
        for i in range(1, 101):  # 假设每个字母有100个样本
            audio_path = f'data/{label}/{label}_{i}.wav'  # 构造音频文件路径
            if os.path.exists(audio_path):  # 如果文件存在
                audio_files_by_label[label].append(audio_path)

    # 对每个字母的音频数据训练模型
    for label, audio_files in audio_files_by_label.items():
        print(f"Training model for label '{label}'...")
        model = train_gmm_hmm(audio_files, label)  # 调用train_gmm_hmm函数训练模型
        if model is not None:
            print(f"Model for label '{label}' trained and saved.")  # 成功训练并保存模型
        else:
            print(f"Failed to train model for label '{label}'.")  # 训练失败

def record_audio(output_path, duration=3, samplerate=16000):
    """
    录制音频并保存为WAV文件。
    """
    try:
        print("Recording... Speak now!")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')  # 录音
        sd.wait()  # 等待录音完成

        # 将音频数据保存为WAV文件
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # 设置单声道
            wf.setsampwidth(2)  # 设置采样宽度（16位）
            wf.setframerate(samplerate)  # 设置采样率
            wf.writeframes(audio_data.tobytes())  # 写入音频数据

        print(f"Recording saved to {output_path}")  # 提示保存路径
        return True  # 返回成功
    except Exception as e:
        print(f"Error during recording: {e}")  # 如果发生错误，打印错误信息
        return False  # 返回失败


def user_interface():
    """
    用户界面：提供语音录入和文件选择功能。
    """
    def predict_from_recording():
        """
        从录制的音频中进行预测。
        """
        output_path = "recorded_audio.wav"
        success = record_audio(output_path)  # 录制音频
        if success:
            result = predict_with_gmm_hmm(models, output_path)  # 使用GMM-HMM模型进行预测
            if result:
                messagebox.showinfo("Prediction Result", f"The predicted character is: {result}")  # 显示预测结果
            else:
                messagebox.showerror("Error", "Prediction failed.")  # 预测失败，显示错误信息

    def predict_from_file():
        """
        从文件中选择音频进行预测。
        """
        audio_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])  # 弹出文件选择框
        if audio_path:
            result = predict_with_gmm_hmm(models, audio_path)  # 使用GMM-HMM模型进行预测
            if result:
                messagebox.showinfo("Prediction Result", f"The predicted character is: {result}")  # 显示预测结果
            else:
                messagebox.showerror("Error", "Prediction failed.")  # 预测失败，显示错误信息

    # 创建GUI界面
    root = tk.Tk()
    root.title("GMM-HMM Voice Recognition")
    root.geometry("400x200")

    tk.Label(root, text="GMM-HMM Voice Recognition", font=("Arial", 14)).pack(pady=10)  # 显示标题

    record_button = tk.Button(root, text="Record Voice", command=predict_from_recording, height=2, width=20)  # 录制按钮
    record_button.pack(pady=10)

    file_button = tk.Button(root, text="Select Audio File", command=predict_from_file, height=2, width=20)  # 文件选择按钮
    file_button.pack(pady=10)

    root.mainloop()  # 启动GUI界面


if __name__ == "__main__":
    train_models()  # 训练所有模型
    models = {label: load_model(label) for label in 'abcdefghijklmnopqrstuvwxyz'}  # 加载所有模型
    if any(models.values()):  # 检查是否有有效模型
        user_interface()  # 启动用户界面
    else:
        print("No models trained. Please train models first.")  # 如果没有加载到模型，提示用户训练模型