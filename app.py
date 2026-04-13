from flask import Flask, request, jsonify, render_template
from haiku_generator import build_system

app = Flask(__name__)

system = build_system(
    config_path="data/haiku_config_v2.json",
    evaluator_model_path="model/haiku_evaluator_model.pth",
    evaluator_vocab_path="model/haiku_vocab.json",
    eval_thr=0.77
)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/generate")
@app.post("/api/generate")
def generate():
    data = request.get_json(silent=True) or {}
    kigo = (data.get("kigo") or "").strip()

    if not kigo:
        return jsonify({"ok": False, "error": "kigo is required"}), 400

    num_candidates = int(data.get("num_candidates", 400))
    top_k = int(data.get("top_k", 10))
    num_candidates = max(1, min(num_candidates, 500))
    top_k = max(1, min(top_k, 50))

    rc = data.get("return_candidates", False)
    if isinstance(rc, str):
        rc = rc.strip().lower() in ("true", "1", "yes", "y", "on")
    else:
        rc = bool(rc)

    print(f"DEBUG: kigo={kigo}, num_candidates={num_candidates}, return_candidates={rc}, top_k={top_k}")
    

    try:
        result = system.generate(
            kigo=kigo,
            num_candidates=num_candidates,
            top_k=top_k,
            return_candidates=rc
        )
        print("DEBUG result:", result)
        return jsonify(result)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"ok": False, "error": str(e)}), 400
    
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)