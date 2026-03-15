"""
frontend/streamlit_app.py – ANPR System with RBAC
Role-aware pages:
  All roles       → Login / Register / Detect / My History / My Stats
  Operational+    → Team History / Export CSV
  Admin           → Admin Dashboard / User Management
"""

import os, io, time
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from PIL import Image

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")

st.set_page_config(page_title="ANPR System", page_icon="🚗", layout="wide")

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b, #0f172a);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] button {
    background-color: #334155 !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] button:hover {
    background-color: #475569 !important;
    border-color: #64748b !important;
    transform: translateY(-1px);
}
[data-testid="stSidebar"] * {
    color: #f1f5f9;
}
.role-badge-normal_user      { background:#dbeafe; color:#1e40af; border-radius:99px;
                                padding:2px 12px; font-size:.78rem; font-weight:700; }
.role-badge-operational_user { background:#d1fae5; color:#065f46; border-radius:99px;
                                padding:2px 12px; font-size:.78rem; font-weight:700; }
.role-badge-admin            { background:#fef3c7; color:#92400e; border-radius:99px;
                                padding:2px 12px; font-size:.78rem; font-weight:700; }
.plate-badge { background:#fef08a; border:3px solid #ca8a04; border-radius:8px;
               padding:8px 24px; font-size:2rem; font-weight:800;
               font-family:'Courier New',monospace; letter-spacing:4px;
               color:#1c1917; display:inline-block; }
.stat-box { text-align:center; background:#f8fafc; border-radius:10px;
            padding:18px 10px; border:1px solid #e2e8f0; }
.stat-val { font-size:2.2rem; font-weight:800; color:#0f172a; }
.stat-lbl { font-size:.8rem; color:#64748b; }
</style>""", unsafe_allow_html=True)


# ── Session helpers ────────────────────────────────────────────────────────────

def _init():
    for k, v in {"token": None, "user": None, "page": "login",
                 "last_result": None}.items():
        st.session_state.setdefault(k, v)

def _logged_in(): return bool(st.session_state.get("token"))
def _user():      return st.session_state.get("user") or {}
def _role():      return _user().get("role", "")
def _perms():     return set(_user().get("permissions", []))
def _has(perm):   return perm in _perms()
def _hdrs():      return {"Authorization": f"Bearer {st.session_state['token']}"}

def _role_badge(role: str) -> str:
    labels = {"normal_user": "Normal User", "operational_user": "Operational", "admin": "Admin"}
    return f'<span class="role-badge-{role}">{labels.get(role, role)}</span>'


# ── API helpers ────────────────────────────────────────────────────────────────

def api(method, endpoint, **kwargs) -> dict:
    url = f"{API_BASE}{endpoint}"
    try:
        r = getattr(requests, method)(url, timeout=60, **kwargs)
        if r.headers.get("Content-Type","").startswith("text/csv"):
            return {"__csv__": r.content, "success": True}
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "❌ Cannot reach API server. Is Flask running?"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def gget(ep, **kw):  return api("get",    ep, headers=_hdrs(), **kw)
def ppost(ep, **kw): return api("post",   ep, headers=_hdrs(), **kw)
def ppatch(ep,**kw): return api("patch",  ep, headers=_hdrs(), **kw)
def ddelete(ep,**kw):return api("delete", ep, headers=_hdrs(), **kw)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## 🚗 ANPR System")
        if not _logged_in():
            if st.button("Login",    use_container_width=True): _goto("login")
            if st.button("Register", use_container_width=True): _goto("register")
            return

        u = _user()
        st.markdown(f"**{u.get('username','')}**")
        st.markdown(_role_badge(_role()), unsafe_allow_html=True)
        st.caption(u.get("email",""))
        st.markdown("---")

        # All roles
        st.markdown("**Detection**")
        _nav("📸 Detect Plate",  "detect")
        _nav("📋 My History",    "history")
        _nav("📊 My Stats",      "stats")

        # Operational +
        if _has("view_team") or _has("view_all"):
            st.markdown("---")
            st.markdown("**Team**")
            if _has("view_team") or _has("view_all"):
                _nav("👥 Team History",  "team_history")
            if _has("export_csv"):
                _nav("⬇️  Export CSV",   "export")

        # Admin
        if _has("system_stats"):
            st.markdown("---")
            st.markdown("**Admin**")
            _nav("🖥️  Dashboard",       "admin_dashboard")
            _nav("👤 User Management",  "admin_users")

        st.markdown("---")
        _nav("💬 AI Assistant", "assistant")
        st.markdown("---")
        if st.button("🔓 Logout", use_container_width=True):
            st.session_state.update({"token": None, "user": None,
                                     "page": "login", "last_result": None})
            st.rerun()

def _nav(label, page):
    if st.button(label, use_container_width=True):
        _goto(page)

def _goto(page):
    st.session_state["page"] = page
    st.rerun()


# ── Auth pages ─────────────────────────────────────────────────────────────────

def page_login():
    _, col, _ = st.columns([1,1.5,1])
    with col:
        st.markdown("# 🔑 Login")
        ident = st.text_input("Username or Email")
        pwd   = st.text_input("Password", type="password")
        if st.button("Sign In", use_container_width=True, type="primary"):
            if not ident or not pwd:
                st.error("Fill all fields.")
            else:
                r = api("post", "/auth/login", json={"username": ident, "password": pwd})
                if r.get("success"):
                    st.session_state.update({"token": r["access_token"],
                                             "user": r["user"], "page": "detect"})
                    st.rerun()
                else:
                    st.error(r.get("message"))
        st.markdown(""); 
        if st.button("Create account →"): _goto("register")


def page_register():
    _, col, _ = st.columns([1,1.5,1])
    with col:
        st.markdown("# 📝 Register")
        st.caption("New accounts start as **Normal User**. An admin can promote you later.")
        un = st.text_input("Username")
        em = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        pw2= st.text_input("Confirm Password", type="password")
        if st.button("Register", use_container_width=True, type="primary"):
            if not all([un,em,pw,pw2]):
                st.error("Fill all fields.")
            elif pw != pw2:
                st.error("Passwords don't match.")
            elif len(pw) < 8:
                st.error("Password must be ≥ 8 characters.")
            else:
                r = api("post", "/auth/register",
                        json={"username": un, "email": em, "password": pw})
                if r.get("success"):
                    st.session_state.update({"token": r["access_token"],
                                             "user": r["user"], "page": "detect"})
                    st.rerun()
                else:
                    st.error(r.get("message"))
        if st.button("Already have an account? Sign In →"): _goto("login")


# ── Detect page ────────────────────────────────────────────────────────────────

def page_detect():
    st.markdown("## 📸 Detect Number Plate")
    limit_mb = _user().get("role") in ("operational_user","admin") and 64 or 16
    st.caption(f"Upload limit for your role: **{limit_mb} MB**")

    uploaded = st.file_uploader("Upload vehicle image",
                                type=["jpg","jpeg","png","bmp","webp"])
    col_img, col_res = st.columns([1.2,1])

    if uploaded:
        img_bytes = uploaded.read()
        with col_img:
            st.image(img_bytes, caption="Uploaded", use_container_width=True)

        with col_res:
            if st.button("🔍 Run ANPR", use_container_width=True, type="primary"):
                with st.spinner("Running YOLOv8 + EasyOCR…"):
                    r = api("post", "/detect",
                            files={"image": (uploaded.name, img_bytes, uploaded.type)},
                            headers=_hdrs())
                st.session_state["last_result"] = r

            result = st.session_state.get("last_result")
            if result:
                if result.get("success"):
                    d = result["result"]
                    st.markdown("### ✅ Result")
                    st.markdown(f'<div class="plate-badge">{d["plate_text"]}</div>',
                                unsafe_allow_html=True)
                    st.markdown("")
                    st.progress(d["yolo_confidence"], text=f"YOLO  {d['yolo_confidence']:.1%}")
                    st.progress(d["ocr_confidence"],  text=f"OCR   {d['ocr_confidence']:.1%}")
                    st.caption(f"Saved as record #{d['record_id']}")

                    # Fetch annotated image
                    fn = os.path.basename(d["image_path"])
                    ann = api("get", f"/image/{fn}", headers=_hdrs())
                    if isinstance(ann, dict) and not ann.get("success"):
                        pass
                    else:
                        ann_r = requests.get(f"{API_BASE}/image/{fn}",
                                             headers=_hdrs(), timeout=30)
                        if ann_r.status_code == 200:
                            with col_img:
                                st.image(ann_r.content, caption="Annotated",
                                         use_container_width=True)
                else:
                    st.warning(result.get("message","Detection failed."))


# ── History helpers ────────────────────────────────────────────────────────────

def _render_history_table(records: list, show_user: bool = False):
    if not records:
        st.info("No records found.")
        return

    cols = (["👤 User"] if show_user else []) + ["🔢 Plate","YOLO","OCR","🕒 Time","🔍"]
    widths = ([1.5] if show_user else []) + [2, 1, 1, 2.2, 0.5]
    header_cols = st.columns(widths)
    for hc, lbl in zip(header_cols, cols):
        hc.markdown(f"**{lbl}**")
    st.markdown("---")

    for rec in records:
        plate = rec.get("plate_text") or "No text detected"
        yc    = rec.get("yolo_confidence")
        oc    = rec.get("ocr_confidence")
        ts    = rec.get("timestamp","")[:16].replace("T"," ")
        row   = st.columns(widths)
        idx   = 0
        if show_user:
            row[idx].markdown(f"`{rec.get('username','?')}`"); idx+=1
        row[idx].markdown(f"**`{plate}`**");        idx+=1
        row[idx].markdown(f"{yc:.1%}" if yc else "—"); idx+=1
        row[idx].markdown(f"{oc:.1%}" if oc else "—"); idx+=1
        row[idx].markdown(f"{ts}");                  idx+=1
        with row[idx]:
            key = f"exp_{rec['id']}"
            if st.button("👁", key=f"btn_{rec['id']}"):
                st.session_state[key] = not st.session_state.get(key, False)
        if st.session_state.get(f"exp_{rec['id']}"):
            with st.expander(f"Record #{rec['id']}", expanded=True):
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.write(f"**Plate:** `{plate}`")
                    st.write(f"**YOLO:** {f'{yc:.4f}' if yc else '—'}")
                    st.write(f"**OCR:** {f'{oc:.4f}' if oc else '—'}")
                    st.write(f"**Time:** {ts} UTC")
                    if show_user:
                        st.write(f"**User:** {rec.get('username','?')}")
                    # Admin delete button
                    if _has("delete_any"):
                        if st.button(f"🗑 Delete Record #{rec['id']}",
                                     key=f"del_{rec['id']}"):
                            dr = ddelete(f"/history/{rec['id']}")
                            if dr.get("success"):
                                st.success("Deleted.")
                                st.rerun()
                            else:
                                st.error(dr.get("message"))
                with dc2:
                    if rec.get("image_path"):
                        fn = os.path.basename(rec["image_path"])
                        ann_r = requests.get(f"{API_BASE}/image/{fn}",
                                             headers=_hdrs(), timeout=30)
                        if ann_r.status_code == 200:
                            st.image(ann_r.content, use_container_width=True)
        st.markdown('<hr style="margin:3px 0;border-color:#f1f5f9">',
                    unsafe_allow_html=True)


def _history_page(title: str, scope_param: str = ""):
    st.markdown(f"## {title}")
    col_l, col_r = st.columns([3,1])
    with col_r:
        per_page = st.selectbox("Per page", [10,20,50], index=1, key=f"pp_{scope_param}")
    with col_l:
        page = st.number_input("Page", min_value=1, value=1, step=1, key=f"pg_{scope_param}")

    data = gget("/history", params={"page": page, "per_page": per_page})
    if not data.get("success"):
        st.error(data.get("message"))
        return

    pag     = data.get("pagination", {})
    records = data.get("records", [])
    scope   = data.get("scope","own")
    show_user = scope in ("team","all")

    st.caption(f"Scope: **{scope}** · {pag.get('total',0)} records total")
    _render_history_table(records, show_user=show_user)


# ── Individual pages ───────────────────────────────────────────────────────────

def page_history():
    _history_page("📋 My History", "own")


def page_team_history():
    _history_page("👥 Team History", "team")


def page_stats():
    st.markdown("## 📊 My Stats")
    d = gget("/stats")
    if not d.get("success"):
        st.error(d.get("message")); return

    s     = d["stats"]
    scope = d.get("scope","personal")
    cols  = st.columns(3)
    for col, label, val in [
        (cols[0], "Total Scans",    s["total_detections"]),
        (cols[1], "Successful",     s["successful_detections"]),
        (cols[2], "Failed",         s["failed_detections"]),
    ]:
        col.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div>'
                     f'<div class="stat-lbl">{label}</div></div>',
                     unsafe_allow_html=True)


def page_export():
    st.markdown("## ⬇️ Export History CSV")
    scope = "all records" if _has("view_all") else "team records"
    st.info(f"This will export **{scope}** based on your role.")
    if st.button("Generate & Download CSV", type="primary"):
        with st.spinner("Preparing CSV…"):
            r = gget("/export/csv")
        if r.get("success") and "__csv__" in r:
            st.download_button(
                label="📥 Download anpr_history.csv",
                data=r["__csv__"],
                file_name="anpr_history.csv",
                mime="text/csv",
            )
        else:
            st.error(r.get("message","Export failed."))


def page_assistant():
    st.markdown("## 💬 AI Assistant")
    st.caption("Ask questions about the ANPR system, your history, or database stats!")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm the ANPR AI Assistant. How can I help you today?"}
        ]
        
    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle new user input
    if prompt := st.chat_input("Ask me anything..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI Response via the Backend API
        with st.chat_message("assistant"):
            with st.spinner("Thinking (and maybe querying the DB)..."):
                try:
                    # Pass the question to our new backend /chat endpoint
                    r = ppost("/chat", json={
                        "message": prompt,
                        "history": st.session_state.messages[:-1] # Exclude the current prompt
                    })
                    
                    if r.get("success"):
                        answer_text = r.get("answer", "No answer provided.")
                        st.markdown(answer_text)
                        
                        # Optionally surface the SQL query in a spoiler/expander if ran
                        sql_ran = r.get("sql_executed")
                        if sql_ran:
                            with st.expander("Show Database Query used"):
                                st.code(sql_ran, language="sql")
                                
                        st.session_state.messages.append({"role": "assistant", "content": answer_text})
                    else:
                        st.error(r.get("message", "Unknown error from backend API."))
                except Exception as e:
                    st.error(f"Error connecting to API server: {e}")


# ── Admin pages ────────────────────────────────────────────────────────────────

def page_admin_dashboard():
    st.markdown("## 🖥️ Admin Dashboard")
    d = gget("/admin/dashboard")
    if not d.get("success"):
        st.error(d.get("message")); return

    db = d["dashboard"]

    # Top metrics
    m1,m2,m3,m4,m5 = st.columns(5)
    for col, lbl, val in [
        (m1, "Total Scans",    db["total_detections"]),
        (m2, "Successful",     db["successful_detections"]),
        (m3, "Failed",         db["failed_detections"]),
        (m4, "Success Rate",   f"{db['success_rate']}%"),
        (m5, "Total Users",    db["total_users"]),
    ]:
        col.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div>'
                     f'<div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 👥 Users by Role")
        for role, count in db["users_by_role"].items():
            label = {"normal_user":"Normal","operational_user":"Operational","admin":"Admin"}.get(role,role)
            pct   = count / max(db["total_users"],1)
            st.markdown(f"**{label}**: {count}")
            st.progress(pct)

    with col_r:
        st.markdown("### 🏆 Top Plates Detected")
        for i, item in enumerate(db["top_plates"], 1):
            st.markdown(f"`{i}.` **`{item['plate']}`** — {item['hits']} hits")

    st.markdown("### 🔥 Most Active Users")
    for item in db["top_users"]:
        st.markdown(
            f"**{item['username']}** "
            f"{_role_badge(item['role'])} "
            f"— {item['scans']} scans",
            unsafe_allow_html=True,
        )


def page_admin_users():
    st.markdown("## 👤 User Management")

    # Filters
    fc1, fc2, _ = st.columns([1,1,2])
    role_filter = fc1.selectbox("Filter by Role",
                                ["All","normal_user","operational_user","admin"])
    page        = fc2.number_input("Page", min_value=1, value=1, step=1)

    params = {"page": page, "per_page": 20}
    if role_filter != "All":
        params["role"] = role_filter

    data = gget("/admin/users", params=params)
    if not data.get("success"):
        st.error(data.get("message")); return

    users = data.get("users", [])
    pag   = data.get("pagination", {})
    st.caption(f"{pag.get('total',0)} users total")

    if not users:
        st.info("No users found."); return

    # Header
    hc = st.columns([2,2,1.5,1,1,0.8,0.8,0.8])
    for col,lbl in zip(hc,["Username","Email","Role","Scans","Status","Promote","Toggle","Delete"]):
        col.markdown(f"**{lbl}**")
    st.markdown("---")

    ROLES = ["normal_user","operational_user","admin"]

    for u in users:
        uid  = u["id"]
        is_me = uid == _user().get("id")
        c = st.columns([2,2,1.5,1,1,0.8,0.8,0.8])
        c[0].markdown(f"**{u['username']}**")
        c[1].markdown(f"<small>{u['email']}</small>", unsafe_allow_html=True)
        c[2].markdown(_role_badge(u["role"]), unsafe_allow_html=True)
        c[3].markdown(str(u.get("total_detections",0)))
        c[4].markdown("✅ Active" if u["is_active"] else "🔴 Inactive")

        # Role selector + promote button (skip self)
        with c[5]:
            if not is_me:
                new_r = st.selectbox("", ROLES,
                                     index=ROLES.index(u["role"]) if u["role"] in ROLES else 0,
                                     key=f"role_sel_{uid}", label_visibility="collapsed")
                if st.button("Set", key=f"set_role_{uid}"):
                    r = ppatch(f"/admin/users/{uid}/role", json={"role": new_r})
                    st.toast(r.get("message","Done"), icon="✅" if r.get("success") else "❌")
                    st.rerun()
            else:
                st.caption("(you)")

        # Activate / Deactivate
        with c[6]:
            if not is_me:
                if u["is_active"]:
                    if st.button("🔴", key=f"deact_{uid}", help="Deactivate"):
                        ppatch(f"/admin/users/{uid}/deactivate")
                        st.rerun()
                else:
                    if st.button("✅", key=f"act_{uid}", help="Activate"):
                        ppatch(f"/admin/users/{uid}/activate")
                        st.rerun()

        # Delete
        with c[7]:
            if not is_me:
                if st.button("🗑", key=f"del_user_{uid}", help="Delete user"):
                    st.session_state[f"confirm_del_{uid}"] = True
                if st.session_state.get(f"confirm_del_{uid}"):
                    if st.button("⚠️ Confirm", key=f"conf_{uid}"):
                        ddelete(f"/admin/users/{uid}")
                        st.session_state.pop(f"confirm_del_{uid}", None)
                        st.rerun()

        st.markdown('<hr style="margin:3px 0;border-color:#f1f5f9">',
                    unsafe_allow_html=True)


# ── Router ─────────────────────────────────────────────────────────────────────

PAGE_MAP = {
    "detect":           page_detect,
    "history":          page_history,
    "team_history":     page_team_history,
    "stats":            page_stats,
    "export":           page_export,
    "admin_dashboard":  page_admin_dashboard,
    "admin_users":      page_admin_users,
    "assistant":        page_assistant,
}

PROTECTED = {
    "team_history": "view_team",
    "export":       "export_csv",
    "admin_dashboard": "system_stats",
    "admin_users":  "manage_roles",
}


def main():
    _init()
    sidebar()
    page = st.session_state["page"]

    if not _logged_in():
        page_register() if page == "register" else page_login()
        return

    # Permission guard
    required = PROTECTED.get(page)
    if required and not _has(required):
        st.error(f"🔒 You don't have permission to access this page.")
        st.info(f"Required: `{required}` · Your role: `{_role()}`")
        return

    PAGE_MAP.get(page, page_detect)()


if __name__ == "__main__":
    main()