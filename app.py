import re
import streamlit as st
from ingestion_module import ingest_story
from rag_module import generate_response
from summarizer import summarize_memory

from memory_store import add_memory_to_ltm, retrieve_memories_from_ltm, _MEM_STORE

from datetime import datetime, timedelta
import time

try:
    from decay_scheduler import DecayEngine, DecayConfig
    DECAY_AVAILABLE = True
except ImportError:
    DECAY_AVAILABLE = False
    print("decay_scheduler.py not found - decay features disabled")

BOT_NAME = "Story Recall Bot"

st.set_page_config(page_title=BOT_NAME, layout="wide")
st.title(f"🧠 {BOT_NAME}")
st.caption("AI-powered story memory with recall, reinforcement, and decay.")

try:
    from encoder import generate_embedding
    print("app.py: warming up embedding model...")
    _ = generate_embedding("warmup")
    print("app.py: embedding warm-up done.")
except Exception as e:
    print("app.py: embedding warm-up failed:", e)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("assistant", f"Hello! I'm {BOT_NAME}. Tell me a story or ask me what I remember.")
    ]


class InMemoryBackendWrapper:
    """Makes _MEM_STORE compatible with DecayEngine."""
    
    def get_all(self):
        return _MEM_STORE.copy()
    
    def update(self, memory_id, updates):
        for mem in _MEM_STORE:
            if mem.get("id") == memory_id:
                if "metadata" in updates:
                    mem.setdefault("metadata", {}).update(updates["metadata"])
                return True
        return False
    
    def delete(self, memory_id):
        for i, mem in enumerate(_MEM_STORE):
            if mem.get("id") == memory_id:
                _MEM_STORE.pop(i)
                return True
        return False
    
    def get(self, memory_id):
        for mem in _MEM_STORE:
            if mem.get("id") == memory_id:
                return mem
        return None


def run_manual_decay(decay_rate=0.05, forget_threshold=0.20, 
                     consolidate_threshold=0.40, min_age_days=1):
    """Execute manual decay cycle with given parameters."""
    
    if not DECAY_AVAILABLE:
        st.sidebar.error("⚠️ decay_scheduler.py not found")
        return
    
    if len(_MEM_STORE) == 0:
        st.sidebar.warning("📝 No memories to decay")
        return
    
    config = DecayConfig(
        decay_rate=decay_rate,
        forget_threshold=forget_threshold,
        consolidation_threshold=consolidate_threshold,
        min_age_days=min_age_days
    )
    
    backend = InMemoryBackendWrapper()
    
    from summarizer import summarize_memory
    
    engine = DecayEngine(
        config=config,
        backend=backend,
        summarizer_fn=summarize_memory
    )
    
    with st.sidebar:
        with st.spinner("🧠 Running decay cycle..."):
            start_time = time.time()
            stats = engine.run_decay_cycle()
            elapsed = time.time() - start_time
        
        st.success(f"Decay complete in {elapsed:.2f}s")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Updated", stats.get("updated", 0))
        with col2:
            st.metric("Consolidated", stats.get("consolidated", 0))
        with col3:
            st.metric("Forgotten", stats.get("forgotten", 0))


def generate_test_memories(count, age_days):
    """Generate test memories with specified age for decay testing."""
    
    test_stories = [
        "A brave knight fought a dragon in the mountains.",
        "The scientist discovered a cure for the mysterious illness.",
        "Children played in the park on a sunny afternoon.",
        "The detective solved the case using clever deduction.",
        "An astronaut explored a distant planet's surface.",
    ]
    
    from encoder import generate_embedding
    
    for i in range(min(count, len(test_stories))):
        story = test_stories[i]
        
        old_time = datetime.utcnow() - timedelta(days=age_days)
        
        memory_dict = {
            "id": f"test_{i}_{int(time.time())}",
            "content_summary": story,
            "raw": story,
            "embedding": generate_embedding(story),
            "metadata": {
                "importance_score": 0.5 + (i * 0.05),
                "access_count": 1,
                "timestamp": old_time.isoformat(),
                "test_generated": True
            }
        }
        
        add_memory_to_ltm(memory_dict)
    
    st.sidebar.success(f"✅ Generated {count} test memories ({age_days} days old)")
    st.rerun()


class LocalLLMWrapper:
    """
    Minimal wrapper that ingestion_module can call as llm_chain.
    It simply calls the summarizer and returns a compact JSON string.
    """
    def __call__(self, prompt_text: str):
        try:
            summary = summarize_memory(prompt_text, min_len=20, max_len=80)
        except Exception:
            summary = (prompt_text[:200] + "...") if len(prompt_text) > 200 else prompt_text

        fallback_json = {
            "title": (prompt_text[:60] + "...") if len(prompt_text) > 60 else prompt_text,
            "content_summary": summary,
            "characters": [],
            "setting": None,
            "key_events": [],
            "themes": [],
            "importance_score": 0.5
        }
        import json
        return json.dumps(fallback_json)

local_llm = LocalLLMWrapper()

STORY_RE = re.compile(r'^\s*(story|narrative)\s*[:\-]?\s*(.+)$', flags=re.IGNORECASE)


with st.sidebar:
    st.markdown("---")
    st.subheader("🧠 Memory Management")
    
    total_memories = len(_MEM_STORE)
    st.metric("Total Memories", total_memories)
    
    if total_memories > 0:
        scores = [m.get("metadata", {}).get("importance_score", 0.5) for m in _MEM_STORE]
        avg_score = sum(scores) / len(scores) if scores else 0
        st.metric("Avg Importance", f"{avg_score:.3f}")
        
        oldest_age = 0
        for m in _MEM_STORE:
            ts_str = m.get("metadata", {}).get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    age = (datetime.utcnow() - ts.replace(tzinfo=None)).days
                    oldest_age = max(oldest_age, age)
                except:
                    pass
        st.metric("Oldest Memory", f"{oldest_age} days")
    
    st.markdown("---")
    
    if DECAY_AVAILABLE and total_memories > 0:
        st.markdown("### 🧹 Run Decay Cycle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🐌 Normal", use_container_width=True, help="Paper default: rate=0.05"):
                run_manual_decay(decay_rate=0.05, forget_threshold=0.20)
        
        with col2:
            if st.button("⚡ Fast", use_container_width=True, help="3x faster for testing"):
                run_manual_decay(decay_rate=0.15, forget_threshold=0.30)
        
        with st.expander("⚙️ Custom Settings"):
            decay_rate = st.slider("Decay Rate", 0.01, 0.20, 0.05, 0.01)
            forget_threshold = st.slider("Forget Threshold", 0.05, 0.50, 0.20, 0.05)
            consolidate_threshold = st.slider("Consolidation Threshold", 0.20, 0.70, 0.40, 0.05)
            
            if st.button("🎯 Run Custom", type="primary", use_container_width=True):
                run_manual_decay(
                    decay_rate=decay_rate,
                    forget_threshold=forget_threshold,
                    consolidate_threshold=consolidate_threshold
                )
    
    elif not DECAY_AVAILABLE:
        st.warning("⚠️ Decay unavailable\n\nInstall: `pip install apscheduler`")
    else:
        st.info("📝 Add some stories to enable decay testing")
    
    st.markdown("---")
    st.subheader("🧪 Testing Tools")
    
    with st.expander("Generate Test Memories"):
        num_stories = st.number_input("Number of test stories", 1, 5, 3)
        age_days = st.number_input("Age (days ago)", 0, 30, 7)
        
        if st.button("Generate Test Data"):
            generate_test_memories(num_stories, age_days)


for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

if user_input := st.chat_input("Ask a question or share a story..."):
    print(f"\n[USER] {user_input}")

    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    m = STORY_RE.match(user_input)
    if m:
        story_text = m.group(2).strip()
        print(f"[INGESTION DETECTED] story_text='{story_text[:80]}'")

        try:
            memory_obj = ingest_story(story_text, llm_chain=local_llm)
        except TypeError:
            memory_obj = ingest_story(story_text)

        if memory_obj:
            if hasattr(memory_obj, "dict"):
                mem_dict = memory_obj.dict()
            elif isinstance(memory_obj, dict):
                mem_dict = memory_obj
            else:
                mem_dict = {
                    "id": getattr(memory_obj, "id", None),
                    "content_summary": getattr(memory_obj, "content_summary", None),
                    "raw": getattr(memory_obj, "raw", None),
                    "embedding": getattr(memory_obj, "embedding", None),
                    "metadata": getattr(memory_obj, "metadata", {}),
                }

            try:
                mid = add_memory_to_ltm(mem_dict)
                summary_text = mem_dict.get("content_summary") or mem_dict.get("title") or ""
                print(f"[STORY STORED] id={mid} | summary='{summary_text[:100]}'")

                probe_query = " ".join(summary_text.split()[:6]) if summary_text else "test"
                probe_res = retrieve_memories_from_ltm(probe_query, top_k=5)
                probe_count = len(probe_res)
                print(f"[PROBE] query='{probe_query}' → {probe_count} result(s)")

                bot_reply = (
                    f"Stored memory id={mid}. Summary: {summary_text}\n\n"
                    f"Probe query: '{probe_query}' → {probe_count} result(s)."
                )
            except Exception as e:
                print(f"[ERROR] storing memory: {e}")
                bot_reply = f"Stored locally but failed to record ID: {e}. Summary preserved."
        else:
            print("[INGESTION FAILED] No memory_obj returned.")
            bot_reply = "Sorry – I couldn't ingest that story right now."

    else:
        stm_context = [f"{role.capitalize()}: {text}" for role, text in st.session_state.chat_history]
        retrieval_context = {"query": user_input}
        print(f"[RETRIEVE] query='{user_input}'")
        try:
            bot_reply = generate_response(retrieval_context, stm_context)
        except TypeError:
            bot_reply = generate_response(user_input, stm_context)
        print(f"[ANSWER] {bot_reply.splitlines()[0][:120]}...")

    st.session_state.chat_history.append(("assistant", bot_reply))
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    try:
        with st.sidebar.expander("Debug: memory store", expanded=False):
            st.write(f"Memory store size: {len(_MEM_STORE)}")
            if len(_MEM_STORE) > 0:
                for e in _MEM_STORE[-5:][::-1]:
                    st.write({
                        "id": e.get("id"),
                        "summary": (e.get("content_summary") or "")[:200],
                        "embedding_len": len(e.get("embedding")) if e.get("embedding") else None,
                        "metadata": e.get("metadata", {})
                    })
    except Exception as e:
        print(f"[DEBUG PANEL ERROR] {e}")