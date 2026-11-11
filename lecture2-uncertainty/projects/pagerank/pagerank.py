import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    corpus_links = corpus
    n_corpus_links = len(corpus_links)

    page_links = corpus[page]
    n_page_links = len(page_links)

    if n_page_links > 0:            
        for link in corpus_links:
            distribution[link] = (1 - damping_factor) / n_corpus_links

        for link in page_links:
            distribution[link] += (damping_factor) * (1 / n_page_links)
    else:
        for link in corpus_links:
            distribution[link] = 1 / n_corpus_links

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    random_page = random.choice(list(corpus))
    transitions = transition_model(corpus, random_page, damping_factor)
    
    current_keys = list(transitions.keys())
    current_probs = list(transitions.values())

    pagerank = {item: 0 for item in corpus}

    for _ in range(n):
        next_page = random.choices(current_keys, weights=current_probs)[0]
        pagerank[next_page] += 1 / n
        transitions = transition_model(corpus, next_page, damping_factor)
        current_keys = list(transitions.keys())
        current_probs = list(transitions.values())

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    corpus_links = corpus
    n_corpus_links = len(corpus_links)
    initial_rank = 1 / n_corpus_links
    pagerank = {item: initial_rank for item in corpus_links}
    pagerank_prev = {item: 0 for item in corpus_links} 
    current_page_rank_change = initial_rank

    while current_page_rank_change > 0.001:
        for page in corpus_links:
            rank = 0
            for key, links in corpus_links.items():
                if len(links) == 0:
                    rank += pagerank[key] / n_corpus_links
                if page in links:
                    rank += pagerank[key] / len(links)
            pagerank[page] = (1 - damping_factor) + (damping_factor * rank)
        
        current_page_rank_change = max([abs(pagerank_prev[key] - item) 
                                        for key, item in pagerank.items()])
        pagerank_prev = {key: item for key, item in pagerank.items()}

    for key in pagerank:
        pagerank[key] = pagerank[key] / n_corpus_links

    return pagerank


if __name__ == "__main__":
    main()
