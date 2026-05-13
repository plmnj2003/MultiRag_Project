# RunPod에서 MultiRag_Project 실행 가이드

## 1. RunPod 인스턴스 설정

### 권장 템플릿
- **Template**: `RunPod Pytorch 2.x` 또는 `RunPod TensorFlow`
- **GPU**: RTX 3090 / A5000 / A6000 (VRAM 24GB+ 권장, 로컬 모델 사용 시)
- **Container Disk**: 30GB 이상
- **Volume Disk**: 20GB 이상

> Claude API를 사용하면 GPU 없이 **CPU 인스턴스**도 가능합니다.

---

## 2. 프로젝트 업로드

### 방법 A: git clone (GitHub에 올린 경우)
```bash
git clone https://github.com/your-username/MultiRag_Project.git
cd MultiRag_Project
```

### 방법 B: RunPod 파일 업로드 (UI)
RunPod 대시보드 → Files 탭 → 폴더 통째로 업로드

### 방법 C: rsync (SSH 연결 후)
```bash
rsync -avz /Users/sunmiyang/Desktop/MultiRag_Project/ \
  root@<runpod-ip>:<pod-port>/workspace/MultiRag_Project/
```

---

## 3. 환경 설정

```bash
cd /workspace/MultiRag_Project

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. API 키 설정

```bash
# .env 파일 편집
nano .env
```

`.env` 내용:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
CLAUDE_MODEL=claude-sonnet-4-6
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./outputs/chroma_db
```

---

## 5. 실행

```bash
# 가상환경 활성화
source venv/bin/activate

# 메인 파이프라인 실행
python main.py
```

---

## 6. MultimodalQA 실제 데이터셋 사용 (선택)

```bash
# MultimodalQA 데이터셋 다운로드
cd data/multimodalqa

# 공식 GitHub에서 데이터 다운로드
pip install gdown
python -c "
import urllib.request
url = 'https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset/MMQA_dev.jsonl.gz'
urllib.request.urlretrieve(url, 'MMQA_dev.jsonl.gz')
import gzip, shutil
with gzip.open('MMQA_dev.jsonl.gz', 'rb') as f_in:
    with open('MMQA_dev.jsonl', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print('다운로드 완료')
"
```

---

## 7. Jupyter Notebook으로 실행 (선택)

```bash
pip install jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

RunPod 대시보드에서 포트 8888 연결 후 브라우저에서 접속.

---

## 8. 비용 절감 팁

| 설정 | 비용 | 비고 |
|------|------|------|
| CPU 인스턴스 | 저렴 | Claude API 사용 시 GPU 불필요 |
| GPU 인스턴스 | 보통 | 로컬 임베딩 모델 가속 |
| Spot 인스턴스 | 50%↓ | 중단될 수 있으므로 체크포인트 필요 |

---

## 9. 트러블슈팅

### `ModuleNotFoundError` 발생 시
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 메모리 부족 시
`.env`에서 작은 임베딩 모델로 교체:
```
EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
```

### ANTHROPIC_API_KEY 오류 시
```bash
echo $ANTHROPIC_API_KEY  # 환경변수 확인
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx  # 직접 설정
```
