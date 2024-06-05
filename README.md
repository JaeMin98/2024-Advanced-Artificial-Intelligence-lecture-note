
# Advanced Artificial Intelligence (인공지능 특강) 환경 설치 방법
Spring Semester, 2024

## 설치 가이드 (Windows 기준)

### 1. SWIG 설치
- [SWIG 3.0.12 다운로드 링크](http://www.swig.org/download.html)
- 다운로드 후, `swig.exe`가 포함된 SWIG 디렉토리를 시스템 환경 변수에 추가

### 2. Microsoft Visual C++ 14.0 설치
- Visual Studio Community 또는 Build Tools를 통해 설치 가능
- 설치 방법은 [공식 문서](https://visualstudio.microsoft.com/ko/downloads/) 참조

### 3. Conda 환경 생성
- Anaconda 또는 Miniconda가 설치되어 있어야 합니다.
- 다음 명령어를 실행하여 Conda 환경을 생성합니다:
  ```sh
  conda create -n gymenv python=3.8.10
  ```

### 4. pip 업그레이드
- 다음 명령어를 실행하여 pip를 최신 버전으로 업그레이드합니다:
  ```sh
  python -m pip install --upgrade pip
  ```

### 5. 필요 라이브러리 설치
- 프로젝트에 필요한 라이브러리를 설치하기 위해 `requirements.txt` 파일을 사용합니다:
  ```sh
  pip install -r requirements.txt
  ```

---

추가적으로, 과제와 강의에 대한 정보는 아래 링크를 통해 확인할 수 있습니다:

- [강의 홈페이지](http://link.koreatech.ac.kr/lecture/2024/advanced_ai)
- [Homework 1](https://www.dropbox.com/scl/fi/iyat052w8oous1p148f9g/HW_1.pdf?rlkey=qggwkbwvkz7ihbutnk247nvrq&e=1&dl=0)
- [Homework 2](https://www.dropbox.com/scl/fi/wdas1lo3l3bsx1hhp2x6z/HW_2.pdf?rlkey=8atvaerw5mydoitb4a34x5mne&e=1&dl=0)

---