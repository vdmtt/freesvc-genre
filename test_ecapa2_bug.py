# """Do rieng hai van de cua ECAPA2:
#
#   A. sai sample rate  : train+infer deu dua 24k vao encoder von la 16k
#   B. lech shape       : train dua (B,1,T), infer dua (1,T)
#
# Chay tu thu muc goc repo:  python test_ecapa2_bug.py
# """
# import itertools
#
# import librosa
# import torch
# import torch.nn.functional as F
# import torchaudio
#
# from models.speaker_encoders import ECAPA2SpeakerEncoder16k
#
# # ---------------------------------------------------------------- cau hinh
# # Sua 3 dong nay cho khop data cua ban.
# SINGER_A = [
#     "dataset_custom/audio24k/english/Alec Benjamin_00010002.wav",
#     "dataset_custom/audio24k/english/Alec Benjamin_00010015.wav",
# ]
# SINGER_B = [
#     "dataset_custom/audio24k/english/Harrison Storm_00120081.wav",
#     "dataset_custom/audio24k/english/Harrison Storm_00120085.wav",
# ]
# SR = 24000  # = data.sampling_rate
#
# enc = ECAPA2SpeakerEncoder16k(device="cpu")
# enc.eval()
#
#
# # ------------------------------------------------------- 3 cach goi encoder
# @torch.no_grad()
# def emb_train(path):
#     """(1, 1, T) @ 24k — dung nhu get_spk_emb luc train."""
#     w, _ = librosa.load(path, sr=SR, duration=5.0)
#     return enc(torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0)).flatten()
#
#
# @torch.no_grad()
# def emb_infer(path):
#     """(1, T) @ 24k — dung nhu inference_online_spk.py."""
#     w, _ = librosa.load(path, sr=SR,duration=5.0)
#     return enc(torch.from_numpy(w).float().unsqueeze(0)).flatten()
#
#
# @torch.no_grad()
# def emb_patch(path):
#     """(1, T) @ 16k — sau khi ap patch."""
#     w, _ = librosa.load(path, sr=SR, duration=5.0)
#     y = torch.from_numpy(w).float().unsqueeze(0)
#     return enc(torchaudio.functional.resample(y, SR, 16000)).flatten()
#
#
# def cos(a, b):
#     return F.cosine_similarity(a, b, dim=0).item()
#
#
# # ------------------------------------------------------------------- kiem tra
# print("=" * 66)
# print("0. SHAPE")
# print("=" * 66)
# g_train = emb_train(SINGER_A[0])
# g_infer = emb_infer(SINGER_A[0])
# g_patch = emb_patch(SINGER_A[0])
# print(f"  train  (1,1,T)@24k -> {tuple(g_train.shape)}")
# print(f"  infer  (1,T)  @24k -> {tuple(g_infer.shape)}")
# print(f"  patch  (1,T)  @16k -> {tuple(g_patch.shape)}")
#
# if g_train.shape != g_infer.shape:
#     print("\n  !! shape KHAC NHAU giua train va infer — day la loi nghiem trong,")
#     print("     khong chi la lech gia tri. Dung lai va xu ly truoc.")
#     raise SystemExit(1)
#
#
# print()
# print("=" * 66)
# print("B. LECH SHAPE  (cung 24k, chi khac cach goi)")
# print("=" * 66)
# print("  1.0000 = ECAPA2 coi hai shape nhu nhau -> model cu infer KHONG lech")
# print("  thap hon = model cu that su infer lech so voi luc train\n")
# for p in SINGER_A + SINGER_B:
#     ten = p.rsplit("/", 1)[-1]
#     print(f"  {cos(emb_train(p), emb_infer(p)):.4f}   {ten}")
#
#
# print()
# print("=" * 66)
# print("A. SAI SAMPLE RATE  (kha nang phan biet ca si)")
# print("=" * 66)
# print("  'khoang cach' cang lon cang tot. Rong hon o dong patch -> nen sua.\n")
#
# pairs_same = list(itertools.combinations(SINGER_A, 2)) + \
#              list(itertools.combinations(SINGER_B, 2))
# pairs_diff = [(a, b) for a in SINGER_A for b in SINGER_B]
#
# for ten, f in [("hien tai (24k)", emb_train), ("patch    (16k)", emb_patch)]:
#     cache = {p: f(p) for p in SINGER_A + SINGER_B}
#     same = sum(cos(cache[a], cache[b]) for a, b in pairs_same) / len(pairs_same)
#     diff = sum(cos(cache[a], cache[b]) for a, b in pairs_diff) / len(pairs_diff)
#     print(f"  {ten}:  cung {same:.4f} | khac {diff:.4f} | khoang cach {same - diff:.4f}")
#
# print()
# print("  (chi 2 ca si — nhieu con lon. Them ca si vao SINGER_A/B de chac chan hon.)")
import time, torch, librosa
from models.speaker_encoders import ECAPA2SpeakerEncoder16k

enc = ECAPA2SpeakerEncoder16k(device='cpu'); enc.eval()
w, _ = librosa.load("dataset_custom/audio24k/english/Alec Benjamin_00010002.wav",
                    sr=24000, duration=5.0)
print("do dai:", len(w)/24000, "giay", flush=True)

t = time.time()
with torch.no_grad():
    g = enc(torch.from_numpy(w).float().unsqueeze(0))
print("shape:", tuple(g.shape), "| thoi gian:", round(time.time()-t, 2), "s")