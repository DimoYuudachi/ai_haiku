// DOM要素を取得
const kigoInput = document.getElementById('kigo');
const generateBtn = document.getElementById('generateBtn');
const btnText = document.getElementById('btnText');
const errorDiv = document.getElementById('error');
const errorMsg = document.getElementById('errorMsg');
const resultDiv = document.getElementById('result');
const haikuDiv = document.getElementById('haiku');

// 季語をセット
function setKigo(value) {
    kigoInput.value = value;
}

// ローディング状態を切り替え
function setLoading(loading) {
    generateBtn.disabled = loading;
    kigoInput.disabled = loading;
    
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(btn => btn.disabled = loading);
    
    btnText.textContent = loading ? '生成中…' : '生成';
}

// エラー表示
function showError(message) {
    errorMsg.textContent = message;
    errorDiv.classList.add('show');
}

// エラー非表示
function hideError() {
    errorDiv.classList.remove('show');
}

// 結果非表示
function hideResult() {
    resultDiv.classList.remove('show');
    haikuDiv.innerHTML = '';
}

// 結果表示
function showResult(data) {
    let list = data.top || data.candidates || [];
    
    if (list.length === 0) {
        showError('俳句が生成できませんでした');
        return;
    }

    const top3 = list.slice(0, 3);
    haikuDiv.innerHTML = '';

    // Top3を表示
    top3.forEach((item, i) => {
        const block = document.createElement('div');
        block.className = 'top-block';

        // 俳句本文
        const lines = item.haiku.split('\n');
        lines.forEach(line => {
            const lineDiv = document.createElement('div');
            lineDiv.className = 'haiku-line';
            lineDiv.textContent = line;
            block.appendChild(lineDiv);
        });

        // スコア表示
        const score = item.evaluator_prob || item.score;
        if (score) {
            const scoreSection = document.createElement('div');
            scoreSection.className = 'score-section';

            const scoreLabel = document.createElement('div');
            scoreLabel.className = 'score-label';
            scoreLabel.textContent = '評価スコア';
            scoreSection.appendChild(scoreLabel);

            const scoreBarContainer = document.createElement('div');
            scoreBarContainer.className = 'score-bar-container';

            const scoreBar = document.createElement('div');
            scoreBar.className = 'score-bar';

            const scoreFill = document.createElement('div');
            scoreFill.className = 'score-fill';
            const percentage = (score * 100).toFixed(1);
            scoreFill.style.width = percentage + '%';
            scoreBar.appendChild(scoreFill);

            const scoreValue = document.createElement('div');
            scoreValue.className = 'score-value';
            scoreValue.textContent = percentage + '%';

            scoreBarContainer.appendChild(scoreBar);
            scoreBarContainer.appendChild(scoreValue);
            scoreSection.appendChild(scoreBarContainer);

            block.appendChild(scoreSection);
        }

        haikuDiv.appendChild(block);
    });

    resultDiv.classList.add('show');
}

// 俳句を生成
async function generateHaiku() {
    const kigo = kigoInput.value.trim();

    if (!kigo) {
        showError('季語を入力してください');
        return;
    }

    hideError();
    hideResult();
    setLoading(true);

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                kigo: kigo,
                num_candidates: 400,
                top_k: 3,
                return_candidates: true
            })
        });

        const data = await response.json();

        if (response.ok) {
            showResult(data);
        } else {
            showError(data.error || '生成に失敗しました');
        }
    } catch (err) {
        showError('エラーが発生しました');
    } finally {
        setLoading(false);
    }
}

// 初期化
kigoInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') generateHaiku();
});