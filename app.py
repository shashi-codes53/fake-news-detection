from flask import Flask, request, jsonify
from flask_cors import CORS
import os, pickle, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)

MODEL_PATH     = "outputs/checkpoints/tfidf_model.pkl"
tfidf_pipeline = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Fake News Detector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a14;color:#e2e8f0;font-family:'Segoe UI',sans-serif;min-height:100vh}
header{background:linear-gradient(135deg,#1a0533,#0a0a14);border-bottom:1px solid #2a2a3d;padding:18px 24px;display:flex;align-items:center;justify-content:space-between}
.logo{display:flex;align-items:center;gap:12px;font-size:20px;font-weight:700;color:#a78bfa}
.logo-icon{width:40px;height:40px;background:#7c3aed;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px}
.badge{background:#1e1040;border:1px solid #a78bfa;color:#a78bfa;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600}
.hero{text-align:center;padding:60px 24px 40px;background:radial-gradient(ellipse at top,#1a0533 0%,transparent 60%)}
.hero h1{font-size:clamp(28px,5vw,52px);font-weight:800;background:linear-gradient(135deg,#fff 30%,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2;margin-bottom:16px}
.hero p{color:#64748b;font-size:16px;max-width:500px;margin:0 auto 32px;line-height:1.6}
.stats{display:flex;justify-content:center;gap:32px;flex-wrap:wrap;margin-bottom:40px}
.stat{text-align:center}
.stat-num{font-size:28px;font-weight:800;color:#a78bfa}
.stat-label{font-size:12px;color:#64748b;margin-top:2px}
.main-card{max-width:720px;margin:0 auto 60px;padding:0 16px}
.card{background:#13131f;border:1px solid #2a2a3d;border-radius:16px;padding:28px}
label{display:block;font-size:13px;font-weight:600;color:#64748b;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em}
textarea{width:100%;min-height:160px;background:#0d0d1a;border:1px solid #2a2a3d;border-radius:10px;color:#e2e8f0;padding:14px;font-size:14px;font-family:inherit;resize:vertical;outline:none;transition:border-color .2s;line-height:1.6}
textarea:focus{border-color:#a78bfa}
textarea::placeholder{color:#64748b}
.char-count{text-align:right;font-size:12px;color:#64748b;margin-top:6px;margin-bottom:16px}
.examples{margin-bottom:20px}
.examples-label{font-size:12px;color:#64748b;margin-bottom:8px}
.example-chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{background:#1a1a2e;border:1px solid #2a2a3d;border-radius:999px;padding:5px 14px;font-size:12px;color:#e2e8f0;cursor:pointer;transition:all .2s}
.chip:hover{border-color:#a78bfa;color:#a78bfa}
.btn-analyze{width:100%;padding:14px;background:linear-gradient(135deg,#7c3aed,#a78bfa);border:none;border-radius:10px;color:white;font-size:16px;font-weight:700;cursor:pointer;transition:opacity .2s,transform .1s;letter-spacing:.02em}
.btn-analyze:hover{opacity:.9}
.btn-analyze:active{transform:scale(.99)}
.btn-analyze:disabled{opacity:.5;cursor:not-allowed}
.loading{display:none;text-align:center;padding:28px;color:#a78bfa;font-size:14px}
.spinner{width:36px;height:36px;border:3px solid #2a2a3d;border-top-color:#a78bfa;border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}
#result{display:none;margin-top:24px;animation:fadeUp .4s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.result-card{border-radius:14px;padding:24px;border:1px solid}
.result-card.fake{background:#1c0a0a;border-color:#ef4444}
.result-card.real{background:#0a1c0a;border-color:#22c55e}
.result-card.unsure{background:#1c1400;border-color:#f59e0b}
.result-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.result-emoji{font-size:36px}
.result-label{font-size:26px;font-weight:800}
.fake-color{color:#ef4444}
.real-color{color:#22c55e}
.unsure-color{color:#f59e0b}
.result-cred{font-size:12px;font-weight:700;padding:4px 12px;border-radius:999px;text-transform:uppercase;letter-spacing:.08em}
.cred-low{background:#3d0a0a;color:#ef4444}
.cred-medlow{background:#3d2200;color:#fb923c}
.cred-medhigh{background:#2d3300;color:#a3e635}
.cred-high{background:#0a3d0a;color:#22c55e}
.conf-label{font-size:12px;color:#64748b;margin-bottom:6px;display:flex;justify-content:space-between}
.bar-wrap{background:#1e1e2e;border-radius:999px;height:10px;overflow:hidden;margin-bottom:16px}
.bar-fill{height:100%;border-radius:999px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.bar-fake{background:linear-gradient(90deg,#b91c1c,#ef4444)}
.bar-real{background:linear-gradient(90deg,#15803d,#22c55e)}
.bar-unsure{background:linear-gradient(90deg,#b45309,#f59e0b)}
.result-message{font-size:14px;color:#94a3b8;line-height:1.6;padding:12px;background:#ffffff08;border-radius:8px}
.how-section{max-width:720px;margin:0 auto 60px;padding:0 16px}
.section-title{font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:20px;text-align:center}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.step-card{background:#13131f;border:1px solid #2a2a3d;border-radius:12px;padding:20px;text-align:center}
.step-icon{font-size:28px;margin-bottom:10px}
.step-title{font-size:14px;font-weight:700;margin-bottom:6px;color:#a78bfa}
.step-desc{font-size:12px;color:#64748b;line-height:1.5}
footer{border-top:1px solid #2a2a3d;padding:24px;text-align:center;color:#64748b;font-size:13px}
footer span{color:#a78bfa;font-weight:600}
.error-box{display:none;margin-top:16px;padding:14px;background:#1c0a0a;border:1px solid #ef4444;border-radius:10px;font-size:13px;color:#fca5a5}
</style>
</head>
<body>
<header>
  <div class="logo"><div class="logo-icon">🔍</div>FakeShield AI</div>
  <div class="badge">LIVE · Powered by ML</div>
</header>
<div class="hero">
  <h1>Detect Fake News<br/>with AI Instantly</h1>
  <p>Paste any news headline or article. Our ML model analyzes language patterns to flag misinformation in real time.</p>
  <div class="stats">
    <div class="stat"><div class="stat-num">89%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat"><div class="stat-num">40K+</div><div class="stat-label">Articles Trained</div></div>
    <div class="stat"><div class="stat-num">&lt;1s</div><div class="stat-label">Detection Speed</div></div>
  </div>
</div>
<div class="main-card">
  <div class="card">
    <label>Paste your news article or headline</label>
    <textarea id="newsInput" placeholder="e.g. Scientists discover breakthrough cancer treatment in clinical trial..."></textarea>
    <div class="char-count"><span id="charCount">0</span> characters</div>
    <div class="examples">
      <div class="examples-label">Try an example →</div>
      <div class="example-chips">
        <div class="chip" onclick="setExample('real1')">📰 Real News</div>
        <div class="chip" onclick="setExample('fake1')">🚨 Fake News</div>
        <div class="chip" onclick="setExample('real2')">🧪 Science Article</div>
        <div class="chip" onclick="setExample('fake2')">⚠️ Conspiracy</div>
      </div>
    </div>
    <button class="btn-analyze" id="analyzeBtn" onclick="analyze()">🔍 &nbsp; Analyze Now</button>
    <div class="loading" id="loading"><div class="spinner"></div>Analyzing language patterns...</div>
    <div class="error-box" id="errorBox"></div>
    <div id="result">
      <div class="result-card" id="resultCard">
        <div class="result-header">
          <div><div class="result-emoji" id="resultEmoji"></div><div class="result-label" id="resultLabel"></div></div>
          <div class="result-cred" id="resultCred"></div>
        </div>
        <div class="conf-label"><span>Fake probability</span><span id="confPct"></span></div>
        <div class="bar-wrap"><div class="bar-fill" id="barFill" style="width:0%"></div></div>
        <div class="result-message" id="resultMsg"></div>
      </div>
    </div>
  </div>
</div>
<div class="how-section">
  <div class="section-title">How It Works</div>
  <div class="steps">
    <div class="step-card"><div class="step-icon">✍️</div><div class="step-title">1. You Paste Text</div><div class="step-desc">Enter any news headline, article, or social media post you want to verify.</div></div>
    <div class="step-card"><div class="step-icon">🧠</div><div class="step-title">2. AI Analyzes</div><div class="step-desc">TF-IDF converts text to vectors. Logistic Regression detects fake patterns.</div></div>
    <div class="step-card"><div class="step-icon">📊</div><div class="step-title">3. See Results</div><div class="step-desc">Get a credibility score, confidence percentage, and actionable advice.</div></div>
  </div>
</div>
<footer>Built by <span>Shashikant Nikam</span> · TF-IDF + Logistic Regression · Flask API</footer>
<script>
const EXAMPLES={
  real1:"Scientists at Johns Hopkins University published research showing combination therapy significantly reduces tumor size in early-stage lung cancer patients, confirmed across three independent clinical trials.",
  fake1:"BREAKING: Government hiding SHOCKING truth about vaccines!!! URGENT share before deleted!!! They are putting microchips in vaccines to control population wake up sheeple!!!",
  real2:"A peer-reviewed study in Nature Medicine found regular aerobic exercise for 30 minutes five times per week reduces cardiovascular disease risk by 35 percent in adults over 50.",
  fake2:"EXPOSED: Illuminati deep state reptilian elites caught shapeshifting on camera!!! Mainstream media covering up the truth!!! Watch before banned by big tech globalists!!!"
};
const textarea=document.getElementById('newsInput');
const charCount=document.getElementById('charCount');
textarea.addEventListener('input',()=>{charCount.textContent=textarea.value.length});
function setExample(key){textarea.value=EXAMPLES[key];charCount.textContent=textarea.value.length;textarea.focus()}
async function analyze(){
  const text=textarea.value.trim();
  if(!text){textarea.focus();return}
  document.getElementById('result').style.display='none';
  document.getElementById('errorBox').style.display='none';
  document.getElementById('loading').style.display='block';
  document.getElementById('analyzeBtn').disabled=true;
  try{
    const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    document.getElementById('loading').style.display='none';
    document.getElementById('analyzeBtn').disabled=false;
    if(!res.ok)throw new Error('Server error '+res.status);
    showResult(await res.json());
  }catch(err){
    document.getElementById('loading').style.display='none';
    document.getElementById('analyzeBtn').disabled=false;
    const box=document.getElementById('errorBox');
    box.style.display='block';
    box.textContent='❌ Error: '+err.message;
  }
}
function showResult(data){
  const conf=data.confidence,isFake=data.label==='FAKE',isUnsure=conf>=0.35&&conf<0.65;
  const card=document.getElementById('resultCard'),emoji=document.getElementById('resultEmoji'),
        label=document.getElementById('resultLabel'),cred=document.getElementById('resultCred'),
        bar=document.getElementById('barFill'),confPct=document.getElementById('confPct'),
        msg=document.getElementById('resultMsg');
  card.className='result-card';bar.className='bar-fill';label.className='result-label';
  if(isUnsure){card.classList.add('unsure');emoji.textContent='⚠️';label.textContent='UNCERTAIN';label.classList.add('unsure-color');bar.classList.add('bar-unsure')}
  else if(isFake){card.classList.add('fake');emoji.textContent='🚨';label.textContent='LIKELY FAKE';label.classList.add('fake-color');bar.classList.add('bar-fake')}
  else{card.classList.add('real');emoji.textContent='✅';label.textContent='LIKELY REAL';label.classList.add('real-color');bar.classList.add('bar-real')}
  const credClass={'LOW':'cred-low','MEDIUM-LOW':'cred-medlow','MEDIUM-HIGH':'cred-medhigh','HIGH':'cred-high'}[data.credibility]||'cred-high';
  cred.textContent='Credibility: '+data.credibility;cred.className='result-cred '+credClass;
  confPct.textContent=Math.round(conf*100)+'%';msg.textContent=data.message;
  document.getElementById('result').style.display='block';
  setTimeout(()=>{bar.style.width=Math.round(conf*100)+'%'},50);
}
textarea.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')analyze()});
</script>
</body>
</html>"""


def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_or_train():
    global tfidf_pipeline
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                tfidf_pipeline = pickle.load(f)
            print("[STARTUP] Model loaded from disk")
            return
    except Exception as e:
        print(f"[STARTUP] Load failed: {e}")

    print("[STARTUP] Training model...")
    texts = [
        "scientists discover new treatment for cancer patients study shows promising results",
        "government announces new policy to improve public healthcare system nationwide",
        "stock market rises on strong economic data quarterly earnings report published",
        "researchers publish peer reviewed findings in nature journal on climate change",
        "president signs new infrastructure bill into law after bipartisan senate approval",
        "university study shows regular exercise significantly improves mental health outcomes",
        "new renewable energy project creates thousands of green jobs across the country",
        "health officials recommend annual flu vaccination for prevention during winter season",
        "local election results officially certified after complete recount by authorities",
        "supreme court issues ruling on landmark civil rights case affecting millions",
        "scientists confirm global temperatures rising due to greenhouse gas emissions data",
        "international trade agreement signed between multiple countries to boost economy",
        "hospital reports breakthrough surgery technique reduces recovery time significantly",
        "city council approves new public transportation project to reduce traffic congestion",
        "technology company releases new product after years of research and development",
        "study finds mediterranean diet linked to lower risk of heart disease in adults",
        "nasa successfully launches new satellite to monitor earth climate patterns data",
        "federal reserve adjusts interest rates in response to inflation economic data report",
        "new education policy aims to improve literacy rates in underprivileged communities",
        "world health organization updates guidelines on nutrition and physical activity today",
        "SHOCKING secret they dont want you to know deep state exposed wake up now share",
        "BREAKING government hiding truth about vaccines dangerous side effects covered up lies",
        "URGENT share before deleted illuminati controls media everything is fake news wake up",
        "celebrities caught in massive scandal cover up exposed truth finally revealed today",
        "miracle cure doctors hate this one weird trick cures cancer instantly at home free",
        "you wont believe alien bodies government hiding area 51 leaked proof exposed now",
        "FRAUD election stolen millions of votes evidence proof watch before banned censored",
        "they are poisoning water supply chemtrails mind control exposed microchips in vaccines",
        "billionaire elite plan to control world population secret meeting leaked documents now",
        "mainstream media lying to you real truth hidden from public view wake up sheeple",
        "BOMBSHELL whistleblower exposes government plot to microchip citizens forced vaccines",
        "new world order plan revealed globalist agenda to enslave humanity exposed finally now",
        "crisis actor caught on camera fake shooting staged by government gun control agenda",
        "5g towers causing coronavirus spread truth they dont want you to know share now viral",
        "george soros paying protesters to destroy america open borders globalist puppet master",
        "moon landing was faked nasa admits in secret documents leaked by insider whistleblower",
        "fluoride in water lowers iq government deliberately poisoning citizens since 1950s now",
        "deep state planning false flag attack to start world war three imminent warning share",
        "secret cure for all diseases suppressed by big pharma to keep patients paying forever",
        "reptilian shapeshifters control world governments exposed in leaked footage share now",
    ]
    labels = [0]*20 + [1]*20
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0))
    ])
    pipeline.fit(texts, labels)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    tfidf_pipeline = pipeline
    print("[STARTUP] Model trained and ready!")


load_or_train()


@app.route('/', methods=['GET'])
def index():
    return HTML_PAGE


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": tfidf_pipeline is not None})


@app.route('/predict', methods=['POST'])
def predict():
    if tfidf_pipeline is None:
        return jsonify({"error": "Model not ready"}), 503
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Send JSON with key 'text'"}), 400
    raw_text = data['text'].strip()
    if not raw_text:
        return jsonify({"error": "Text is empty"}), 400
    clean        = clean_text(raw_text)
    prob_fake    = float(tfidf_pipeline.predict_proba([clean])[0, 1])
    label        = "FAKE" if prob_fake >= 0.5 else "REAL"
    if prob_fake >= 0.75:   credibility, message = "LOW",         "Strong signs of misinformation."
    elif prob_fake >= 0.5:  credibility, message = "MEDIUM-LOW",  "May contain misleading info. Verify before sharing."
    elif prob_fake >= 0.25: credibility, message = "MEDIUM-HIGH", "Appears mostly credible. Double-check key claims."
    else:                   credibility, message = "HIGH",        "Appears credible based on language patterns."
    return jsonify({"label": label, "confidence": round(prob_fake, 4), "credibility": credibility, "message": message})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
