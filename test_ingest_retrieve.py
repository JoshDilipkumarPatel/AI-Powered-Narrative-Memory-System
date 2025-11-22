import tempfile
import os
from ingestion_module import ingest_story
from memory_store import add_memory_to_ltm, retrieve_memories_from_ltm

"""
Simple harness to ingest a story and run a set of queries through your RAG pipeline.
Place this file next to your project modules and run with the same Python environment.
"""

import sys
import time

from ingestion_module import ingest_story
from memory_store import add_memory_to_ltm, _MEM_STORE, retrieve_memories_from_ltm
from rag_module import generate_response

STORY = """I must warn you… this book is ATROCIOUS! The cover has mud all over it, and one of the pages has sprung a leak. Even some of the order are in the wrong words. It is the worst of the worst, so don’t say I didn’t warn you… Our story begins with an atrocious King, None of his laws ever made any sense. “Tuesday?” he laughed, “There is no such day, and I’ll hear no more of it as long as I am king.” From that day on in the kingdom, Monday was followed by second Monday or first Wednesday. His daughter was an atrocious Princess. She was not fancy and polite like most princesses, despite the princess lessons she took each day. She never brushed her hair. She smelled of month- old bananas, and she was very devious. She even set traps throughout this story. [Illustrations throughout the story will indicate the Princess’ traps] One day the Princess went missing. In her place there was a note that read: “I’ve taken the Princess. You me catch never will.” After reading the note twice to understand it, the King commanded the greatest hero in the land to find his daughter. The greatest hero in the land happened to be an atrocious hero. He was often too frightened to leave his home, and he was also scared by butterflies. However, he had once defeated two caterpillars and a moth in a single battle, and this made him the greatest of all the heroes in the land. The hero rode to the castle of the meanest villain in the kingdom, but the villain, too was atrocious. He brought flowers to old ladies whenever he could. And out of the kindness of his heart, he once built a house for someone who needed it. [The hero] The hero stormed the gate and charged into the dungeon of the villain’s castle. “Surrender the Princess!” he yelled while shaking in his armor. “I don’t have a Princess,” said the villain, “I use this dungeon to bake cookies, but watch your step because there’s a leak down here.” [An example of a Princess trap with words floating in water.] The hero was more confused than usual. “If you didn’t take the Princess, then who did?” asked the hero. “We should go ask the dragon what he thinks.” Mind you, this was an atrocious dragon. He was known for being tidy and wise, not fierce and ferocious like dragons are supposed to be. He brushed his teeth every night, and when he breathed, there was no fire, just minty freshness. The dragon was very upset when they found him. “Someone’s stolen my gold, and they left this note in its place!” he cried. It read: “Your gold is with the Princess! Mine now is it!” “This is odd,” said the hero. “Very odd,” said the villain. “Frmph,” said the dragon who always flossed his teeth when he was upset. “Perhaps the Queen knows something.” The Queen was running around the castle when they arrived. She, actually, was not so atrocious (for it was she who hired the royal scribe who wrote this very book). “My treasures are missing” she cried. “My books and jewels, and my fine silk.” “Someone has taken them and left this atrocious note.” It read: “Your treasure is mine! Do not trying to bother find it.” “Whoever has stolen your treasure has the Princess,” said the hero. “Here are some flowers,” said the villain. “Look!” said the dragon, “there are footprints of dirt on the floor.” They followed the footprints all the way up to the tallest tower in the tallest part of the castle. When they looked inside they found all the gold, books, jewels, and silk, covered in dirt. And sitting on a throne on top of the pile, was the Princess. “Why did you steal these things and make us look for you?” asked the Queen. “I didn’t want to go to my writing lessons!” said the Princess, “Who needs writing?” “You do,” said the Queen. The villain, knight, and dragon all nodded in agreement. The Queen made the Princess return what she had stolen and apologize. From that day on, the King made the Princess go to her lessons six days a week. She was an atrocious student when they began, but she got better each day. It took the dragon hours to brush all the dirt off his coins. The villain developed a taste for soggy cookies, and he baked them all the time. The hero decided it was finally time to battle a butterfly… and he lost. As for the Queen, she insisted that the entire atrocious tale be written inside of one of her muddy books. I warned you this book was atrocious, and now it is over! Perhaps, one day you could write a better one…"""

QUERIES = [
    "Who was atrocious?",
    "What did the King ban in his kingdom?",
    "Who went missing?",
    "Who was commanded to find the Princess?",
    "What was the hero afraid of?",
    "What did the villain do for old ladies?",
    "What did the dragon breathe instead of fire?",
    "Who took the Princess?",
    "Why did the Princess steal the treasures?",
    "What happened to the Princess after she was caught?"
]

def main():
    print("=== Clearing memory store (if needed) ===")
    try:
        _MEM_STORE.clear()
    except Exception:
        pass

    print("=== Ingesting story ===")
    mem_obj = ingest_story(STORY)
    print("INGEST_RESULT:", type(mem_obj), mem_obj if isinstance(mem_obj, dict) else "")
    try:
        mid = add_memory_to_ltm(mem_obj)
        print("ADD RESULT ID:", mid)
    except Exception as e:
        print("add_memory_to_ltm failed:", e)

    print("STORE AFTER:", len(_MEM_STORE))
    print("LAST ENTRY preview:", _MEM_STORE[-1] if len(_MEM_STORE) else None)

    time.sleep(0.5)

    print("\n=== Running queries ===")
    for q in QUERIES:
        print("\nQ:", q)
        try:
            out = generate_response({"query": q}, stm_context=None, llm_chain=None)
            print("A:", out)
        except Exception as e:
            print("generate_response failed:", e)

    try:
        debug_q = "Who went missing?"
        print("\n=== Retrieval debug for:", debug_q)
        res = retrieve_memories_from_ltm(debug_q, top_k=5)
        print("retrieve_memories_from_ltm returned", len(res), "candidates")
        for i, r in enumerate(res, start=1):
            print(f"candidate {i} id={r.get('id')} score={r.get('score')} summary={r.get('content_summary')[:140]}")
    except Exception as e:
        print("retrieve debug failed:", e)


if __name__ == "__main__":
    main()

