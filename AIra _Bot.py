# AIra Bot — Final with Logo + Light/Dark toggle + Deep Purple Dark (D2)
# Single-file Gradio app. Copy & run in Colab or VS Code.
# If running in a fresh Colab environment, uncomment the install line below.

# !pip install --upgrade gradio huggingface_hub pdfplumber python-docx pillow pytesseract

import io, json
from typing import Optional
import gradio as gr
from PIL import Image

# Optional libs
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

# ---------------------------
# Hard-coded credentials (testing)
# ---------------------------
STUDENT_ACC = "STU123"
STUDENT_PWD = "stu@123"
PROF_ACC = "PRO456"
PROF_PWD = "pro@456"

# ---------------------------
# HF model & client
# ---------------------------
HF_MODEL = "ibm-granite/granite-3.2-2b-instruct"
hf_client: Optional[InferenceClient] = None

def init_hf(token: str):
    global hf_client
    if InferenceClient is None:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub")
    hf_client = InferenceClient(token=token)
    return "Hugging Face token set."

def hf_generate_text(prompt: str, max_new_tokens: int = 512):
    global hf_client
    if hf_client is None:
        return "[ERROR] Set Hugging Face token first."
    # try prompt= then inputs= for compatibility
    try:
        res = hf_client.text_generation(model=HF_MODEL, prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95)
    except TypeError:
        try:
            res = hf_client.text_generation(model=HF_MODEL, inputs=prompt, max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95)
        except Exception as e2:
            return f"[HF ERROR] {e2}"
    except Exception as e:
        return f"[HF ERROR] {e}"

    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        return res.get("generated_text") or res.get("text") or json.dumps(res)
    if isinstance(res, list) and len(res) > 0:
        if isinstance(res[0], dict):
            return res[0].get("generated_text") or json.dumps(res[0])
        return str(res[0])
    return str(res)

# ---------------------------
# File extraction helpers
# ---------------------------
def extract_pdf(file_obj):
    if pdfplumber is None:
        return "PDF extraction requires pdfplumber."
    try:
        data = file_obj.read()
        bio = io.BytesIO(data)
        texts = []
        with pdfplumber.open(bio) as pdf:
            for p in pdf.pages:
                texts.append(p.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        return f"[PDF ERROR] {e}"

def extract_docx(file_obj):
    if docx is None:
        return "DOCX extraction requires python-docx."
    try:
        data = file_obj.read()
        tmp = "/tmp/tmp_docx.docx"
        with open(tmp, "wb") as f:
            f.write(data)
        d = docx.Document(tmp)
        return "\n".join([p.text for p in d.paragraphs if p.text.strip()])
    except Exception as e:
        return f"[DOCX ERROR] {e}"

def extract_txt(file_obj):
    try:
        data = file_obj.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return str(data)
    except Exception as e:
        return f"[TXT ERROR] {e}"

def extract_any(file_obj):
    if file_obj is None:
        return ""
    name = file_obj.name.lower()
    if name.endswith(".pdf"): return extract_pdf(file_obj)
    if name.endswith(".docx"): return extract_docx(file_obj)
    if name.endswith(".txt"): return extract_txt(file_obj)
    return "[UNSUPPORTED FILE]"

# ---------------------------
# OCR & image analysis
# ---------------------------
def ocr_from_image(img: Image.Image):
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def analyze_image(img: Image.Image, language: str):
    ocr_text = ocr_from_image(img)
    prompt = f"""You are AIra Bot. Language: {language}.
OCR text:
\"\"\"{ocr_text}\"\"\"
Task:
1) Identify if image is a receipt/invoice/document/photo.
2) If receipt/invoice, extract vendor, dates, amounts, currency, and item lines.
Return structured JSON."""
    return hf_generate_text(prompt, max_new_tokens=450)

# ---------------------------
# NER and clause extraction
# ---------------------------
def do_ner(text: str, language: str):
    prompt = f"Extract named entities (persons, organizations, dates, monetary amounts) from the text. Return JSON only. Language: {language}\n\nText:\n{text}"
    return hf_generate_text(prompt, max_new_tokens=500)

def do_clauses(text: str, language: str):
    prompt = f"Break the following into numbered clauses and provide a one-line simple explanation for each. Return JSON only. Language: {language}\n\n{text}"
    return hf_generate_text(prompt, max_new_tokens=700)

# ---------------------------
# Budget split calculator
# ---------------------------
def calculate_budget_split(balance):
    try:
        bal = float(balance)
    except:
        return "Enter a valid numeric balance."
    bucket_40 = bal * 0.40
    daily_30 = bal * 0.30
    loan_30 = bal * 0.30
    categories = ["Travel", "Future Expenses", "Emergency Funds"]
    each = bucket_40 / len(categories)
    text = f"Total Balance: ₹{bal:,.2f}\n\nBucket List (40%) = ₹{bucket_40:,.2f}\n"
    for c in categories:
        text += f" - {c}: ₹{each:,.2f}\n"
    text += f"\nDaily Use (30%) = ₹{daily_30:,.2f}\nPersonal Loans (30%) = ₹{loan_30:,.2f}"
    return text

# ---------------------------
# Features & examples
# ---------------------------
FEATURE_GROUPS = {
    "Budget Tools": ["Budget Split Calculator"],
    "Savings": ["Quick Budget","Smart Categorize","Spending Alerts","Savings Goals","Savings Challenges","Bucket Transfer"],
    "Investments": ["Starter Investments","Invest Advice","Return Forecast","Market Summary"],
    "Taxes": ["Tax Tips","Analyze Salary Slip","Form16 Analysis"],
    "Documents": ["Extract File Text","Get Entities (NER)","Simplify Clauses"]
}

EXAMPLES = {
    "Budget Split Calculator":"Enter your bank balance and click Calculate (e.g. 10000).",
    "Quick Budget":"I earn ₹22,000. Rent 8000, Food 3000, Travel 1200, Books 500. Suggest a budget.",
    "Smart Categorize":"Cafe - 300; Hostel - 6500; Train - 120; Books - 450; Netflix - 299. Categorize.",
    "Spending Alerts":"Alert me when weekly food spending exceeds ₹1500.",
    "Savings Goals":"I want to save ₹10,000 in 6 months. I can save monthly. Suggest a plan.",
    "Bucket Transfer":"Split 40% to buckets equally: Travel, Future, Emergency.",
    "Starter Investments":"I can invest ₹2,000/month. Recommend beginner-friendly options.",
    "Invest Advice":"I have ₹100,000 to invest for 5 years, moderate risk.",
    "Return Forecast":"Forecast 5-year returns for ₹5,000/month SIP at 10% annual.",
    "Market Summary":"Summarize current market trends briefly.",
    "Tax Tips":"Tax-saving options for salaried 12 LPA in India under 80C/80D.",
    "Analyze Salary Slip":"Basic 50,000; HRA 20,000; PF 6000; TDS 8000. Analyze and suggest proofs.",
    "Form16 Analysis":"Check Form 16 for taxable income and tax deducted accuracy.",
    "Extract File Text":"Upload invoice PDF and extract text.",
    "Get Entities (NER)":"Extract dates and amounts from this bill.",
    "Simplify Clauses":"Upload a contract and break into simple clauses."
}

# ---------------------------
# Pastel & Dark palette mapping
# ---------------------------
PASTEL = {
    "Student": "#FFD6E8",       # pastel pink
    "Professional": "#FFE9D6",  # pastel peach
    "Budget Tools": "#FFF5E6",  # cream
    "Savings": "#DFFFE1",       # green
    "Investments": "#D6EDFF",   # blue
    "Taxes": "#FFF9C4",         # yellow
    "Documents": "#EAD6FF"      # lavender
}

# Deep Purple Dark (D2) palette for dark mode
D2_DARK = {
    "base": "#231933",        # background
    "card": "#2f2244",        # card
    "text": "#F5E9FF",        # off-white text
    "accent": "#F3A6D1",      # pink-lavender accents
    "button_bg": "#2a1b37",   # dark button bg
    "button_text": "#ffdff3"  # button text
}

# Helper: build theme style (background + button colors + logo font)
def make_theme_style(bg_hex: str, mode: str="light"):
    """
    mode: "light" or "dark"
    bg_hex is the pastel color to use for gradient start (light mode).
    For dark mode we'll use D2_DARK values.
    Returns HTML <style> string to inject.
    """
    if mode == "dark":
        b = D2_DARK["base"]
        card = D2_DARK["card"]
        text = D2_DARK["text"]
        accent = D2_DARK["accent"]
        btn_bg = D2_DARK["button_bg"]
        btn_text = D2_DARK["button_text"]
        # body gradient uses dark base and a slightly lighter card color
        style = f"""
        <style id='theme-style'>
          body {{ background: linear-gradient(120deg, {b} 0%, {card} 100%); transition: background 300ms ease; color: {text}; }}
          .main, .sidebar {{ background: rgba(255,255,255,0.03) !important; color: {text} !important; }}
          .btn-special {{ background:{btn_bg} !important; color:{btn_text} !important; border:1px solid rgba(255,255,255,0.06) !important; }}
          .logo-text {{ color: #F49CCF !important; font-family: 'Brush Script MT', 'Brush Script Std', cursive; font-size:20px; }}
          .muted {{ color: rgba(245,233,255,0.9) !important; }}
          .input, .textbox, .dropdown, .button, .file-upload {{ background: rgba(255,255,255,0.02) !important; color: {text} !important; border:1px solid rgba(255,255,255,0.04) !important; }}
        </style>
        """
        return style
    else:
        # light mode: use pastel bg_hex for gentle gradient
        safe = bg_hex if bg_hex.startswith("#") else "#" + bg_hex
        style = f"""
        <style id='theme-style'>
          body {{ background: linear-gradient(120deg, {safe} 0%, #ffffff 100%); transition: background 300ms ease; color: #111827; }}
          .main, .sidebar {{ background: #ffffff !important; color: #111827 !important; }}
          .btn-special {{ background:#4b2b48 !important; color:#fff !important; border: none !important; }}
          .logo-text {{ color: #F06292 !important; font-family: 'Brush Script MT', 'Brush Script Std', cursive; font-size:20px; }}
          .muted {{ color: #6b7280 !important; }}
          .input, .textbox, .dropdown, .button, .file-upload {{ background: #fff !important; color: #111827 !important; border:1px solid #eee !important; }}
        </style>
        """
        return style

# ---------------------------
# Logo SVG (cute line-art robot) - colored rose-pink stroke
# ---------------------------
ROBOT_SVG = """
<svg width="46" height="46" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <g fill="none" stroke="#F49CCF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <rect x="12" y="18" width="40" height="30" rx="6" />
    <circle cx="28" cy="30" r="3.2" fill="#F49CCF"/>
    <circle cx="36" cy="30" r="3.2" fill="#F49CCF"/>
    <path d="M20 48c0 3 6 6 12 6s12-3 12-6" />
    <rect x="28" y="8" width="8" height="6" rx="2" />
    <path d="M14 12 L18 14" />
    <path d="M50 12 L46 14" />
  </g>
</svg>
"""

# ---------------------------
# Build Gradio app
# ---------------------------
def build_app():
    # initial light theme style with neutral bg
    initial_style = make_theme_style("#f6f8fb", mode="light")
    with gr.Blocks(title="AIra Bot — Finance Assistant", css="""
        .sidebar { background:#ffffff; padding:14px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }
        .main { background:#ffffff; padding:16px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }
        .muted { color:#6b7280; }
        .bigtitle { font-size:26px; font-weight:700; color:#1f2937; }
        .logo-row { display:flex; align-items:center; gap:10px; }
        .logo-box { display:flex; align-items:center; gap:8px; }
        .btn-special { border-radius:8px; padding:8px 12px; }
        """) as demo:

        # inject theme/style
        theme_html = gr.HTML(initial_style)

        # small header with logo and brush-calligraphy text
        header_html = gr.HTML(f"""
            <div class="logo-row">
              <div class="logo-box">{ROBOT_SVG}</div>
              <div class="logo-text">AIra Bot</div>
            </div>
        """)

        state = gr.State({"role": None, "logged_in": False, "theme": "light", "category": None})

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes="sidebar"):
                    gr.HTML("<strong>Navigation</strong>")
                    new_chat = gr.Button("➕ New Chat")
                    search_chats = gr.Button("🔍 Search Chats")
                    library_btn = gr.Button("📚 Library")
                    gr.Markdown("---")
                    # Light/Dark toggle as radio
                    theme_choice = gr.Radio(["Light","Dark"], value="Light", label="Theme (Light / Dark)")
                    gr.Markdown("---")
                    token_input = gr.Textbox(type="password", label="Hugging Face Token", placeholder="hf_xxx...")
                    set_token_btn = gr.Button("Set Token", elem_classes="btn-special")
                    token_status = gr.Textbox(label="Token Status", interactive=False, value="Not set")
                    gr.Markdown("---")
                    logout_btn = gr.Button("Logout", visible=False, elem_classes="btn-special")
                    gr.Markdown("<div class='muted'>Tip: set HF token to enable AI features.</div>")

            with gr.Column(scale=3):
                with gr.Column(elem_classes="main"):
                    # header/logo
                    gr.HTML(f"<div style='display:flex;align-items:center;justify-content:space-between'>{ROBOT_SVG}<div class='logo-text' style='font-size:20px'>AIra Bot</div></div>")
                    gr.Markdown("<div class='muted'>Your professional finance assistant</div>")
                    gr.Markdown("---")
                    # Language at top
                    language = gr.Dropdown(["English","Hindi","Telugu","Tamil","Kannada","Malayalam","Bengali","Marathi","Gujarati"],
                                           value="English", label="Language")
                    gr.Markdown("---")
                    # login area
                    gr.Markdown("### Login")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("*Student Login*")
                            stu_acc = gr.Textbox(label="Account Number")
                            stu_pwd = gr.Textbox(type="password", label="Password")
                            stu_login = gr.Button("Login as Student", elem_classes="btn-special")
                        with gr.Column():
                            gr.Markdown("*Professional Login*")
                            pro_acc = gr.Textbox(label="Account Number")
                            pro_pwd = gr.Textbox(type="password", label="Password")
                            pro_login = gr.Button("Login as Professional", elem_classes="btn-special")

                    login_status = gr.Textbox(label="Login Status", interactive=False, value="Not logged in")

                    gr.Markdown("<hr/>")
                    # Features (hidden until login)
                    gr.Markdown("### Features (appear after login)")
                    feature_category = gr.Radio(list(FEATURE_GROUPS.keys()), label="Category", value="Budget Tools", visible=False)
                    feature_dropdown = gr.Dropdown([], label="Feature", visible=False)
                    example_btn = gr.Button("Fill Example", visible=False, elem_classes="btn-special")
                    user_input = gr.Textbox(lines=4, label="Input / Example", visible=False)

                    # Budget controls (inside Budget Tools category)
                    budget_balance = gr.Number(label="Enter Balance (₹)", visible=False)
                    budget_calc_btn = gr.Button("Calculate Budget Split", visible=False, elem_classes="btn-special")
                    budget_result = gr.Textbox(lines=6, label="Budget Result", visible=False)

                    # file / image
                    file_input = gr.File(label="Upload file (PDF/DOCX/TXT)", visible=False)
                    image_input = gr.Image(type="pil", label="Upload image (receipt/photo)", visible=False)

                    run_feature = gr.Button("Run Feature", visible=False, elem_classes="btn-special")
                    feature_output = gr.Textbox(lines=12, label="Result", visible=False)

                    # Placeholder for search panel and library panel (to be defined later)
                    search_panel = gr.Group(visible=False)
                    with search_panel:
                        gr.Markdown("### Search Chats")
                        search_query = gr.Textbox(label="Search query (optional)", placeholder="e.g., budget, loan")
                        search_results = gr.Textbox(label="Results", interactive=False, lines=10)
                        close_search_btn = gr.Button("Close Search")

                    lib_panel = gr.Group(visible=False)
                    with lib_panel:
                        gr.Markdown("### Library")
                        lib_content = gr.Textbox(label="Saved Items", interactive=False, lines=10)
                        close_lib_btn = gr.Button("Close Library")


        # ----------------- Callbacks -----------------

        # Helper to update theme HTML and button styles
        def update_theme_html(selected_theme: str, category: Optional[str], role: Optional[str]):
            # decide base pastel color depending on category or role (priority: category then role)
            bg_hex = "#f6f8fb"
            if category:
                bg_hex = PASTEL.get(category, bg_hex)
            elif role:
                bg_hex = PASTEL.get(role, bg_hex)
            mode = "dark" if selected_theme == "Dark" else "light"
            return make_theme_style(bg_hex, mode)

        # set token
        def set_token_cb(tok):
            if not tok:
                return "Enter token"
            try:
                return init_hf(tok.strip())
            except Exception as e:
                return f"[TOKEN ERROR] {e}"

        set_token_btn.click(set_token_cb, inputs=[token_input], outputs=[token_status])

        # login callbacks: reveal features and set background + show logout
        def student_login_cb(acc, pwd, lang, theme):
            if acc != STUDENT_ACC or pwd != STUDENT_PWD:
                # even on failure, show role-bg to indicate attempt
                bg = update_theme_html(theme, None, "Student")
                return "Invalid student credentials", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), bg, gr.update(visible=False)
            # success
            feats = FEATURE_GROUPS["Budget Tools"]
            bg = update_theme_html(theme, "Budget Tools", "Student")
            return (f"Student logged in ({acc})",
                    gr.update(visible=True),  # show category
                    gr.update(choices=list(FEATURE_GROUPS.keys()), value="Budget Tools", visible=True),  # category radio updated
                    gr.update(choices=feats, value=feats[0], visible=True),  # feature dropdown updated
                    gr.update(visible=True),  # example btn visible
                    bg,
                    gr.update(visible=True))  # show logout

        # outputs: login_status, feature_category visible, feature_category update, feature_dropdown update, example_btn visible, theme_html value, logout visible
        stu_login.click(student_login_cb, inputs=[stu_acc, stu_pwd, language, theme_choice], outputs=[login_status, feature_category, feature_category, feature_dropdown, example_btn, theme_html, logout_btn])

        def prof_login_cb(acc, pwd, lang, theme):
            if acc != PROF_ACC or pwd != PROF_PWD:
                bg = update_theme_html(theme, None, "Professional")
                return "Invalid professional credentials", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), bg, gr.update(visible=False)
            feats = FEATURE_GROUPS["Budget Tools"]
            bg = update_theme_html(theme, "Budget Tools", "Professional")
            return (f"Professional logged in ({acc})",
                    gr.update(visible=True),
                    gr.update(choices=list(FEATURE_GROUPS.keys()), value="Budget Tools", visible=True),
                    gr.update(choices=feats, value=feats[0], visible=True),
                    gr.update(visible=True),
                    bg,
                    gr.update(visible=True))

        pro_login.click(prof_login_cb, inputs=[pro_acc, pro_pwd, language, theme_choice], outputs=[login_status, feature_category, feature_category, feature_dropdown, example_btn, theme_html, logout_btn])

        # When theme toggle is clicked: update theme immediately (use current category & role from state via inputs)
        def theme_change_cb(theme, current_cat, current_role):
            # update background: if category present use it; else use role; else neutral
            return update_theme_html(theme, current_cat, current_role)

        theme_choice.change(theme_change_cb, inputs=[theme_choice, feature_category, gr.State({"role": None})], outputs=[theme_html])

        # When category changes: populate feature dropdown and update background according to category and theme
        def on_category_change(cat, theme):
            feats = FEATURE_GROUPS.get(cat, [])
            bg = update_theme_html(theme, cat, None)
            return gr.update(choices=feats, value=feats[0] if feats else None, visible=True), bg, gr.update(visible=True)

        feature_category.change(on_category_change, inputs=[feature_category, theme_choice], outputs=[feature_dropdown, theme_html, example_btn])

        # When feature selected: show/hide inputs; special-case Budget Split Calculator
        def on_feature_select(feature_name, category_name, theme):
            show_example = gr.update(visible=True)
            show_user_input = gr.update(visible=True)
            show_file = gr.update(visible=False)
            show_image = gr.update(visible=False)
            show_run = gr.update(visible=True)
            show_output = gr.update(visible=True)
            show_budget_balance = gr.update(visible=False)
            show_budget_calc_btn = gr.update(visible=False)
            show_budget_result = gr.update(visible=False)

            file_features = {"Extract File Text","Get Entities (NER)","Simplify Clauses","Analyze Salary Slip","Form16 Analysis"}
            image_features = {"Extract File Text"}

            if category_name == "Budget Tools" and feature_name == "Budget Split Calculator":
                show_example = gr.update(visible=False)
                show_user_input = gr.update(visible=False)
                show_file = gr.update(visible=False)
                show_image = gr.update(visible=False)
                show_run = gr.update(visible=False)
                show_output = gr.update(visible=False)
                show_budget_balance = gr.update(visible=True)
                show_budget_calc_btn = gr.update(visible=True)
                show_budget_result = gr.update(visible=True)
            else:
                if feature_name in file_features:
                    show_file = gr.update(visible=True)
                if feature_name in image_features:
                    show_image = gr.update(visible=True)

            bg = update_theme_html(theme, category_name, None)
            return show_example, show_user_input, show_file, show_image, show_run, show_output, budget_balance, budget_calc_btn, budget_result, bg

        feature_dropdown.change(on_feature_select,
                                inputs=[feature_dropdown, feature_category, theme_choice],
                                outputs=[example_btn, user_input, file_input, image_input, run_feature, feature_output, budget_balance, budget_calc_btn, budget_result, theme_html])

        # Fill example text
        def fill_example_cb(feature_name):
            return gr.update(value=EXAMPLES.get(feature_name, ""), visible=True)

        example_btn.click(fill_example_cb, inputs=[feature_dropdown], outputs=[user_input])

        # Budget calculate
        budget_calc_btn.click(lambda b: calculate_budget_split(b), inputs=[budget_balance], outputs=[budget_result])

        # Run other features
        def run_feature_cb(category, feature_name, text, file, image, lang):
            if hf_client is None:
                return "[ERROR] Set Hugging Face token first."
            if feature_name == "Extract File Text":
                if not file:
                    return "Upload a file (PDF/DOCX/TXT)."
                return extract_any(file)
            if feature_name == "Get Entities (NER)":
                content = extract_any(file) if file else text
                if not content:
                    return "Provide text or upload a file."
                return do_ner(content, lang)
            if feature_name == "Simplify Clauses":
                content = extract_any(file) if file else text
                if not content:
                    return "Provide text or upload a file."
                return do_clauses(content, lang)
            if image is not None:
                return analyze_image(image, lang)
            prompt = f"You are AIra Bot. Category={category}. Feature={feature_name}. Language={lang}.\nUser input:\n{text}\nProvide a concise professional answer."
            return hf_generate_text(prompt, max_new_tokens=700)

        run_feature.click(run_feature_cb, inputs=[feature_category, feature_dropdown, user_input, file_input, image_input, language], outputs=[feature_output])

        # Logout resets UI and theme to neutral
        def logout_cb():
            neutral = make_theme_style("#f6f8fb", mode="light")
            return ("Logged out", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), neutral, gr.update(visible=False))

        logout_btn.click(logout_cb, outputs=[login_status, feature_category, feature_dropdown, example_btn, theme_html, logout_btn])

        # Search chats logic (moved here)
        # Note: 'chat_history' must be accessible, usually as a global or passed in.
        # Assuming 'chat_history' is a global list as shown in kernel state.
        def open_search_panel():
            global chat_history # Declare chat_history as global to access it
            if not chat_history:
                txt = "No saved chats yet."
            else:
                lines = []
                for i, c in enumerate(sorted(chat_history, key=lambda x:x["timestamp"], reverse=True)[:50], start=1):
                    lines.append(f"{i}. id:{c['id']} [{c['role']}] {c['category']}/{c['feature']} — {c['timestamp']}\nQ: {c['user_text'][:140]}")
                txt = "\n\n".join(lines)
            # When opening search, search_panel visible=True, results updated, lib_panel visible=False
            return gr.update(visible=True), gr.update(value=txt), gr.update(visible=False)

        search_chats.click(open_search_panel, outputs=[search_panel, search_results, lib_panel])

    return demo

if _name_ == "_main_":
    # Ensure chat_history and library_list are initialized if this is the entry point
    # In a typical Colab run, these would be initialized from previous cells or implicitly.
    # For robustness, we might want to ensure they exist or are empty lists here.
    if 'chat_history' not in globals():
        chat_history = []
    if 'library_list' not in globals():
        library_list = []
    app = build_app()
    app.launch(debug=True)