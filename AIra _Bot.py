# Personal Finance Guide Chatbot - OPTIMIZED VERSION
# Fast execution with better error handling
# Run in Google Colab - Single cell

!pip install -q gradio huggingface_hub

import gradio as gr
from huggingface_hub import InferenceClient
import json
from datetime import datetime, timedelta

# User Database
USER_DATABASE = {
    "students": {
        "STU001": {"password": "student123", "name": "Alex Kumar", "balance": 15000},
        "STU002": {"password": "study456", "name": "Priya Sharma", "balance": 12000}
    },
    "professionals": {
        "PRO001": {"password": "work123", "name": "Rajesh Patel", "salary": 75000, "balance": 150000},
        "PRO002": {"password": "prof456", "name": "Anita Singh", "salary": 95000, "balance": 200000}
    }
}

# Session state
session_state = {
    "logged_in": False,
    "user_type": None,
    "account_number": None,
    "user_data": None,
    "hf_token": None
}

class FinanceChatbot:
    def __init__(self):
        self.client = None
        self.token_set = False
    
    def set_token(self, token):
        """Initialize with HF token"""
        try:
            if not token or not token.startswith("hf_"):
                return "❌ Invalid token format. Should start with 'hf_'"
            
            self.client = InferenceClient(token=token)
            self.token_set = True
            session_state["hf_token"] = token
            return "✅ Token set successfully! AI features enabled."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_response(self, user_message, user_type, user_data):
        """Generate AI response"""
        if not self.token_set or not self.client:
            return "⚠️ Please set your Hugging Face token first in the Settings."
        
        try:
            # Create system prompt
            if user_type == "student":
                system_prompt = f"""You are a helpful financial advisor for students.
User: {user_data['name']} | Balance: ₹{user_data['balance']:,}

Provide practical advice on budgeting, savings, and student-friendly investments.
Keep responses concise (under 200 words) and encouraging."""
            else:
                system_prompt = f"""You are an expert financial advisor for professionals.
User: {user_data['name']} | Salary: ₹{user_data['salary']:,} | Balance: ₹{user_data['balance']:,}

Provide professional advice on investments, tax planning, and wealth building.
Keep responses detailed but concise (under 250 words)."""

            # Generate response with timeout protection
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = ""
            try:
                for message in self.client.chat_completion(
                    messages=messages,
                    max_tokens=500,  # Reduced for faster response
                    model="ibm-granite/granite-3.3-2b-instruct",
                    stream=True,
                    temperature=0.7
                ):
                    if hasattr(message.choices[0].delta, 'content') and message.choices[0].delta.content:
                        response += message.choices[0].delta.content
                        
                return response.strip() if response else "No response generated. Please try again."
            
            except Exception as stream_error:
                # Fallback: try non-streaming
                try:
                    result = self.client.chat_completion(
                        messages=messages,
                        max_tokens=500,
                        model="ibm-granite/granite-3.3-2b-instruct",
                        temperature=0.7
                    )
                    return result.choices[0].message.content
                except:
                    return f"⚠️ Model timeout. Here's a quick tip instead:\n\n{self.get_quick_tip(user_type, user_message)}"
        
        except Exception as e:
            return f"❌ Error: {str(e)}\n\nPlease verify your token at huggingface.co/settings/tokens"
    
    def get_quick_tip(self, user_type, message):
        """Fallback tips when AI fails"""
        tips = {
            "student": [
                "💡 Start with the 50/30/20 rule: 50% needs, 30% wants, 20% savings",
                "💡 Track expenses daily using a simple notebook or app",
                "💡 Consider opening a recurring deposit (RD) account for disciplined savings",
                "💡 Use student discounts whenever available - they add up!"
            ],
            "professional": [
                "💡 Maximize 80C deductions (₹1.5L) through ELSS, PPF, or insurance",
                "💡 Build an emergency fund covering 6 months of expenses",
                "💡 Diversify investments: 60% equity, 30% debt, 10% gold",
                "💡 Review and rebalance your portfolio quarterly"
            ]
        }
        import random
        return random.choice(tips.get(user_type, tips["student"]))
    
    def get_feature_response(self, feature_name, user_type, user_data):
        """Generate feature responses instantly"""
        responses = {
            "budget_summary": self.generate_budget_summary(user_type, user_data),
            "expense_categorization": self.categorize_expenses(user_type, user_data),
            "savings_goal": self.track_savings_goal(user_type, user_data),
            "bill_reminder": self.get_bill_reminders(user_type),
            "investment_suggestions": self.suggest_investments(user_type, user_data),
            "net_worth": self.calculate_net_worth(user_type, user_data),
            "tax_saving": self.tax_saving_tips(user_type, user_data),
            "subscription_tracker": self.track_subscriptions(user_type),
            "cash_flow": self.predict_cash_flow(user_type, user_data),
        }
        return responses.get(feature_name, "Feature coming soon!")
    
    def generate_budget_summary(self, user_type, user_data):
        if user_type == "student":
            return f"""📊 **Student Budget Summary**

💰 Current Balance: ₹{user_data['balance']:,}

**Recommended Monthly Budget:**
- 🍽️ Food & Groceries: ₹3,000 (30%)
- 🏠 Hostel/Rent: ₹4,000 (40%)
- 📚 Books & Supplies: ₹1,000 (10%)
- 🚌 Travel: ₹800 (8%)
- 🎮 Entertainment: ₹700 (7%)
- 💾 Savings: ₹500 (5%)

**Total:** ₹10,000/month

💡 Try to save at least 10% of any income!"""
        else:
            salary = user_data['salary']
            return f"""📊 **Professional Budget Summary**

💰 Monthly Salary: ₹{salary:,}
💵 Balance: ₹{user_data['balance']:,}

**50/30/20 Rule:**
- 🏠 Essentials (50%): ₹{int(salary*0.5):,}
- 🎯 Wants (30%): ₹{int(salary*0.3):,}
- 💎 Savings (20%): ₹{int(salary*0.2):,}

**Tax-Saving Target:** ₹{int(salary*0.15):,}
Invest in 80C, NPS, ELSS"""
    
    def categorize_expenses(self, user_type, user_data):
        if user_type == "student":
            return """📈 **Expense Categories (Last Month)**

🍽️ Food: ₹3,200 (32%)
🏠 Rent: ₹4,000 (40%)
📚 Books: ₹950 (9.5%)
🚌 Travel: ₹850 (8.5%)
🎮 Entertainment: ₹700 (7%)
📱 Internet: ₹300 (3%)

**Total:** ₹10,000

⚠️ Food spending slightly high - try meal planning!"""
        else:
            return """📈 **Expense Categories (Last Month)**

🏠 Rent: ₹25,000 (33%)
🛒 Groceries: ₹8,000 (11%)
🚗 Travel: ₹6,000 (8%)
💳 EMIs: ₹15,000 (20%)
🍽️ Dining: ₹5,000 (7%)
🎬 Entertainment: ₹3,000 (4%)
👔 Shopping: ₹7,000 (9%)
⚡ Bills: ₹4,000 (5%)
💰 Savings: ₹2,000 (3%)

**Total:** ₹75,000

💡 Increase savings to 15%+"""
    
    def track_savings_goal(self, user_type, user_data):
        if user_type == "student":
            return """🎯 **Savings Goal Tracker**

**Target:** ₹5,000
**Saved:** ₹3,200 (64%)

🟩🟩🟩🟩🟩🟩⬜⬜⬜⬜

**Remaining:** ₹1,800
**Days Left:** 12

💡 Save ₹150/day to reach goal!

🏆 **Challenges:**
- ☑️ No-Spend Monday
- ⬜ Cook 5 meals (3/5)
- ⬜ Walk vs cab (2/7)"""
        else:
            return """🎯 **Savings Goal Tracker**

**Target:** ₹15,000/month
**Saved:** ₹12,000 (80%)

🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜

**Annual Projection:** ₹1,44,000

📊 **Buckets:**
- Emergency: ₹50,000 ✅
- Vacation: ₹25,000 (→₹40,000)
- Home DP: ₹75,000 (→₹2,00,000)"""
    
    def get_bill_reminders(self, user_type):
        if user_type == "student":
            return """🔔 **Bill Reminders**

📅 **This Week:**
- 📱 Mobile - Nov 20 (2 days) - ₹299
- 🌐 Wi-Fi - Nov 22 (4 days) - ₹500

📅 **Next Week:**
- 🏠 Rent - Nov 30 - ₹4,000
- 📺 Netflix - Dec 1 - ₹199

💰 Total upcoming: ₹5,000"""
        else:
            return """🔔 **Bill Reminders**

📅 **Urgent:**
- ⚡ Electricity - Nov 20 (2 days) - ₹2,500
- 💳 Credit Card - Nov 22 (4 days) - ₹15,000 ⚠️

📅 **This Month:**
- 🏠 Rent - Nov 30 - ₹20,000
- 🚗 Car EMI - Dec 1 - ₹12,000

💰 Total: ₹50,300"""
    
    def suggest_investments(self, user_type, user_data):
        if user_type == "student":
            return """💎 **Investment Ideas**

1. **Recurring Deposit**
   - ₹500/month
   - Returns: 6-7%
   - Safe & disciplined

2. **SIP in Index Funds**
   - ₹500/month
   - Nifty 50 funds
   - Long-term growth

3. **Digital Gold**
   - ₹100-500/month
   - Easy to liquidate

4. **PPF**
   - Lock: 15 years
   - Tax-free: 7-8%

💡 Start: RD + SIP = ₹1,000/month"""
        else:
            salary = user_data['salary']
            inv = int(salary * 0.2)
            return f"""💎 **Investment Portfolio**

**Monthly Capacity:** ₹{inv:,} (20%)

🎯 **Allocation:**
1. Equity MF (60%): ₹{int(inv*0.6):,}
2. Debt (20%): ₹{int(inv*0.2):,}
3. Gold (10%): ₹{int(inv*0.1):,}
4. Emergency (10%): ₹{int(inv*0.1):,}

💰 **20-Year Wealth:**
Investment: ₹{inv*12*20:,}
Expected: ₹{int(inv*12*20*2.5):,}

🏆 **Tax Benefits:**
ELSS: ₹46,800/year
NPS: ₹15,600/year"""
    
    def calculate_net_worth(self, user_type, user_data):
        if user_type == "student":
            bal = user_data['balance']
            assets = bal + 5000
            liab = 2000
            nw = assets - liab
            return f"""💰 **Net Worth**

**Assets:** ₹{assets:,}
- Balance: ₹{bal:,}
- Items: ₹5,000

**Liabilities:** ₹{liab:,}

**Net Worth:** ₹{nw:,}

📈 Target: +₹50,000 this year"""
        else:
            sal = user_data['salary']
            bal = user_data['balance']
            assets = bal + (sal * 24)
            liab = sal * 8
            nw = assets - liab
            return f"""💰 **Net Worth**

**Assets:** ₹{assets:,}
- Balance: ₹{bal:,}
- Investments: ₹{sal*20:,}
- Property: ₹{sal*15:,}

**Liabilities:** ₹{liab:,}

**Net Worth:** ₹{nw:,}

📊 Target: ₹{int(nw*1.15):,} (+15%)"""
    
    def tax_saving_tips(self, user_type, user_data):
        if user_type == "student":
            return """💰 **Tax Awareness**

📚 **Basics:**
- <₹2.5L: No tax
- ₹2.5-5L: 5% tax
- Keep receipts!

💡 **Tips:**
- Scholarships = tax-free
- Loan interest deductible
- Learn about 80C, 80D
- Get PAN card early"""
        else:
            sal = user_data['salary']
            ann = sal * 12
            return f"""💰 **Tax-Saving Guide**

**Annual:** ₹{ann:,} (30% slab)

🎯 **Section 80C (₹1.5L):**
Save ₹45,000 in tax

🏥 **Section 80D:**
Health insurance - Save ₹22,500

💼 **Others:**
- NPS 80CCD(1B): Save ₹15,000
- HRA: Based on rent
- Home Loan: ₹2L interest

💰 **Total Savings:** ₹82,500+

📋 Review Form 16, maximize 80C!"""
    
    def track_subscriptions(self, user_type):
        if user_type == "student":
            return """📱 **Subscriptions**

- Netflix: ₹199/mo
- Spotify: ₹119/mo
- Google One: ₹130/mo
- Medium: ₹75/mo

**Total:** ₹523/mo (₹6,276/year)

⚠️ **Save:**
- Share Netflix: -₹100
- Free Spotify: -₹119
- Cancel Medium: -₹75

💰 Potential: -₹294/mo"""
        else:
            return """📱 **Subscriptions**

- Netflix: ₹649
- Prime: ₹1,499/yr
- Spotify: ₹119
- LinkedIn: ₹1,700
- Gym: ₹2,000
- Cloud: ₹205

**Total:** ₹5,771/mo (₹69,252/year)

⚠️ **Optimize:**
- Gym: 8 visits (₹250/visit)
- LinkedIn: Rarely used
- Consolidate cloud

💰 Save: ₹2,199/mo"""
    
    def predict_cash_flow(self, user_type, user_data):
        if user_type == "student":
            bal = user_data['balance']
            daily = 300
            days = int(bal / daily)
            date = (datetime.now() + timedelta(days=days)).strftime('%b %d')
            return f"""📊 **Cash Flow**

**Balance:** ₹{bal:,}
**Daily Spend:** ₹{daily}

📅 **Prediction:**
- Lasts: ~{days} days
- Until: {date}

💡 **Extend:**
Reduce to ₹250/day → +12 days
Keep ₹2,000 emergency buffer"""
        else:
            bal = user_data['balance']
            sal = user_data['salary']
            exp = int(sal * 0.8)
            months = bal / exp
            return f"""📊 **Cash Flow**

**Balance:** ₹{bal:,}
**Monthly Expense:** ₹{exp:,}

📅 **Runway:** {months:.1f} months
{"✅ Healthy (6+ months)" if months >= 6 else "⚠️ Build to 6 months"}

💰 **3-Month Projection:**
M1: ₹{bal + sal - exp:,}
M2: ₹{bal + (sal*2) - (exp*2):,}
M3: ₹{bal + (sal*3) - (exp*3):,}

Annual surplus: ₹{(sal-exp)*12:,}"""

# Initialize
chatbot = FinanceChatbot()

def initialize_chatbot(hf_token):
    return chatbot.set_token(hf_token)

def login(account_number, password):
    global session_state
    
    if account_number in USER_DATABASE["students"]:
        if USER_DATABASE["students"][account_number]["password"] == password:
            session_state["logged_in"] = True
            session_state["user_type"] = "student"
            session_state["account_number"] = account_number
            session_state["user_data"] = USER_DATABASE["students"][account_number]
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                f"✅ Welcome {session_state['user_data']['name']}!"
            )
    
    if account_number in USER_DATABASE["professionals"]:
        if USER_DATABASE["professionals"][account_number]["password"] == password:
            session_state["logged_in"] = True
            session_state["user_type"] = "professional"
            session_state["account_number"] = account_number
            session_state["user_data"] = USER_DATABASE["professionals"][account_number]
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                f"✅ Welcome {session_state['user_data']['name']}!"
            )
    
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "❌ Invalid. Try STU001/student123 or PRO001/work123"
    )

def logout():
    global session_state
    session_state = {
        "logged_in": False,
        "user_type": None,
        "account_number": None,
        "user_data": None,
        "hf_token": None
    }
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        []
    )

def handle_chat(message, history):
    if not session_state["logged_in"]:
        return history + [[message, "⚠️ Please login first."]]
    
    if not chatbot.token_set:
        return history + [[message, "⚠️ Please set your HF token in Settings."]]
    
    response = chatbot.generate_response(
        message,
        session_state["user_type"],
        session_state["user_data"]
    )
    
    history.append([message, response])
    return history

def handle_feature_click(feature_name):
    if not session_state["logged_in"]:
        return [[None, "⚠️ Please login first."]]
    
    response = chatbot.get_feature_response(
        feature_name,
        session_state["user_type"],
        session_state["user_data"]
    )
    
    return [[None, response]]

# Build Interface
with gr.Blocks(theme=gr.themes.Soft(), title="AIra Bot", css="""
    .aira-title {
        color: #FF1493 !important;
        font-size: 48px !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 10px !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .aira-subtitle {
        text-align: center !important;
        color: #666 !important;
        font-size: 18px !important;
    }
    """) as demo:
    gr.HTML("""
    <div class="aira-title">💰 AIra Bot</div>
    <div class="aira-subtitle">AI-Powered Financial Guidance</div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            hf_token_input = gr.Textbox(
                label="Hugging Face Token",
                type="password",
                placeholder="hf_...",
                info="Get from huggingface.co/settings/tokens"
            )
            init_btn = gr.Button("Set Token", variant="primary")
            init_status = gr.Textbox(label="Status", value="⚠️ Token not set", interactive=False)
    
    login_section = gr.Column(visible=True)
    with login_section:
        gr.Markdown("## 🔐 Login")
        gr.Markdown("**Demo:** STU001/student123 or PRO001/work123")
        with gr.Row():
            account_input = gr.Textbox(label="Account", placeholder="STU001")
            password_input = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login", variant="primary")
        login_status = gr.Textbox(label="Status", interactive=False)
    
    student_portal = gr.Column(visible=False)
    with student_portal:
        gr.Markdown("## 🎓 Student Portal")
        
        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                chatbot_s = gr.Chatbot(height=350)
                msg_s = gr.Textbox(placeholder="Ask about budgeting, savings...")
                with gr.Row():
                    send_s = gr.Button("Send", variant="primary")
                    clear_s = gr.Button("Clear")
            
            with gr.Tab("📊 Features"):
                with gr.Row():
                    gr.Button("📊 Budget").click(lambda: handle_feature_click("budget_summary"), outputs=chatbot_s)
                    gr.Button("📈 Expenses").click(lambda: handle_feature_click("expense_categorization"), outputs=chatbot_s)
                    gr.Button("🎯 Goals").click(lambda: handle_feature_click("savings_goal"), outputs=chatbot_s)
                with gr.Row():
                    gr.Button("🔔 Bills").click(lambda: handle_feature_click("bill_reminder"), outputs=chatbot_s)
                    gr.Button("💎 Invest").click(lambda: handle_feature_click("investment_suggestions"), outputs=chatbot_s)
                    gr.Button("💰 Net Worth").click(lambda: handle_feature_click("net_worth"), outputs=chatbot_s)
        
        logout_s = gr.Button("Logout", variant="stop")
    
    prof_portal = gr.Column(visible=False)
    with prof_portal:
        gr.Markdown("## 💼 Professional Portal")
        
        with gr.Tabs():
            with gr.Tab("💬 Chat"):
                chatbot_p = gr.Chatbot(height=350)
                msg_p = gr.Textbox(placeholder="Ask about investments, taxes...")
                with gr.Row():
                    send_p = gr.Button("Send", variant="primary")
                    clear_p = gr.Button("Clear")
            
            with gr.Tab("📊 Features"):
                with gr.Row():
                    gr.Button("📊 Budget").click(lambda: handle_feature_click("budget_summary"), outputs=chatbot_p)
                    gr.Button("📈 Expenses").click(lambda: handle_feature_click("expense_categorization"), outputs=chatbot_p)
                    gr.Button("🎯 Goals").click(lambda: handle_feature_click("savings_goal"), outputs=chatbot_p)
                with gr.Row():
                    gr.Button("💰 Tax Tips").click(lambda: handle_feature_click("tax_saving"), outputs=chatbot_p)
                    gr.Button("💎 Portfolio").click(lambda: handle_feature_click("investment_suggestions"), outputs=chatbot_p)
                    gr.Button("📊 Cash Flow").click(lambda: handle_feature_click("cash_flow"), outputs=chatbot_p)
        
        logout_p = gr.Button("Logout", variant="stop")
    
    # Events
    init_btn.click(initialize_chatbot, inputs=[hf_token_input], outputs=[init_status])
    login_btn.click(login, inputs=[account_input, password_input], outputs=[login_section, student_portal, prof_portal, login_status])
    
    send_s.click(handle_chat, inputs=[msg_s, chatbot_s], outputs=[chatbot_s]).then(lambda: "", outputs=[msg_s])
    msg_s.submit(handle_chat, inputs=[msg_s, chatbot_s], outputs=[chatbot_s]).then(lambda: "", outputs=[msg_s])
    clear_s.click(lambda: [], outputs=[chatbot_s])
    logout_s.click(logout, outputs=[login_section, student_portal, prof_portal, login_status, chatbot_s])
    
    send_p.click(handle_chat, inputs=[msg_p, chatbot_p], outputs=[chatbot_p]).then(lambda: "", outputs=[msg_p])
    msg_p.submit(handle_chat, inputs=[msg_p, chatbot_p], outputs=[chatbot_p]).then(lambda: "", outputs=[msg_p])
    clear_p.click(lambda: [], outputs=[chatbot_p])
    logout_p.click(logout, outputs=[login_section, student_portal, prof_portal, login_status, chatbot_p])

demo.launch(debug=True, share=True)