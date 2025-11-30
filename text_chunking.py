def chunk(text: str, CHUNK_SIZE: int, CHUNK_OVERLAP: int) -> list[str]:
    """
    Split large text into overlapping word-based chunks.

    Args:
        text (str): The full document text to split.
        CHUNK_SIZE (int): Target number of words per chunk.
        CHUNK_OVERLAP (int): Number of words to overlap between chunks.

    Returns:
        list[str]: Ordered list of text chunks.
    """

    # input validation
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    if CHUNK_SIZE <= 0:
        raise ValueError("chunk_size must be > 0")
    if CHUNK_OVERLAP < 0:
        raise ValueError("overlap must be >= 0")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("overlap must be smaller than chunk_size")

    # normalise white spaces
    normalised = " ".join(text.split())
    if not normalised: #if empty string, then the pdf has no chars
        return []

    words = normalised.split()
    no_words = len(words)

    # if too short, return as a single chunk
    if no_words <= CHUNK_SIZE:
        return [normalised]
    
    #sliding window approach
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < no_words:
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]

        if not chunk_words: #defensive guardrail incase earlier code ever changed
            break
        
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)

        start += step

    return chunks


if __name__ == "__main__":
    print("=== Manual Tests for chunk() ===\n")

    # ---------------------------------------------------
    # Test 1: Very short text (should return 1 chunk)
    # ---------------------------------------------------
    text1 = "Hello world, this is a short sentence."
    chunks1 = chunk(text1, CHUNK_SIZE=50, CHUNK_OVERLAP=10)
    print("Test 1: Short text")
    print("Input:", text1)
    print("Output chunks:", len(chunks1))
    for i, c in enumerate(chunks1):
        print(f"  Chunk {i}: {c}\n")
    print("-" * 70)

    # ---------------------------------------------------
    # Test 2: Medium text (~1200 words)
    # ---------------------------------------------------
    text2 = " ".join([f"word{i}" for i in range(1200)])  # ~1200 words
    chunks2 = chunk(text2, CHUNK_SIZE=500, CHUNK_OVERLAP=100)
    print("Test 2: Medium text (~1200 words)")
    print("Total chunks:", len(chunks2))
    for i, c in enumerate(chunks2):
        preview = " ".join(c.split()[:10])
        print(f"  Chunk {i} (first 10 words): {preview} ...")
    print()
    print("-" * 70)

    # ---------------------------------------------------
    # Test 3: Whitespace normalization
    # ---------------------------------------------------
    text3 = "This   text\n\nhas   lots    of\n\n irregular\t spacing.   "
    chunks3 = chunk(text3, CHUNK_SIZE=10, CHUNK_OVERLAP=3)
    print("Test 3: Whitespace normalization")
    print("Input raw:", repr(text3))
    print("Normalized:", " ".join(text3.split()))
    print("Output chunks:", len(chunks3))
    for i, c in enumerate(chunks3):
        print(f"  Chunk {i}: {c}")
    print()
    print("-" * 70)

    # ---------------------------------------------------
    # Test 4: Exactly equal to chunk size
    # ---------------------------------------------------
    text4 = " ".join([f"token{i}" for i in range(500)])  # exactly 500 words
    chunks4 = chunk(text4, CHUNK_SIZE=500, CHUNK_OVERLAP=100)
    print("Test 4: Exact chunk size")
    print("Total chunks:", len(chunks4))
    for i, c in enumerate(chunks4):
        preview = " ".join(c.split()[:10])
        print(f"  Chunk {i}: {preview} ...")
    print()
    print("-" * 70)

    # ---------------------------------------------------
    # Test 5: Overlap visual example
    # ---------------------------------------------------
    text5 = " ".join([f"token{i}" for i in range(30)])  # 30 small tokens
    chunks5 = chunk(text5, CHUNK_SIZE=10, CHUNK_OVERLAP=3)
    print("Test 5: Overlap demonstration")
    print("Input words: token0 token1 ... token29")
    print("Total chunks:", len(chunks5))
    for i, c in enumerate(chunks5):
        print(f"  Chunk {i}: {c}")
    print("-" * 70)

    # ---------------------------------------------------
    # Test 6: Edge case – chunk_size smaller than text, but small overlap
    # ---------------------------------------------------
    text6 = " ".join([f"alpha{i}" for i in range(60)])
    chunks6 = chunk(text6, CHUNK_SIZE=15, CHUNK_OVERLAP=2)
    print("Test 6: Small chunk size, small overlap")
    print("Total chunks:", len(chunks6))
    for i, c in enumerate(chunks6):
        print(f"  Chunk {i}: {c}")
    print("-" * 70)

    # ---------------------------------------------------
    # Test 7: Empty string
    # ---------------------------------------------------
    text7 = "     \n\t   "
    chunks7 = chunk(text7, CHUNK_SIZE=100, CHUNK_OVERLAP=20)
    print("Test 7: Empty/whitespace-only string")
    print("Output chunks:", len(chunks7), "(should be 0)")
    print("-" * 70)