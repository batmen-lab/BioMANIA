"""
Author: Zhengyuan Dong
Date Created: May 29, 2024
Last Modified: May 29, 2024
Description: Download the closed issues and comments from the scanpy repository
Requirement: pip install PyGithub
Usage: 
python -m src.dataloader.download_issues --LIB scanpy --token {GITHUB_TOKEN}
Email: zydong122@gmail.com
"""

from github import Github
import json
import os
import argparse
from tqdm import tqdm
import time
from ..configs.model_config import get_all_variable_from_cheatsheet
def fetch_issues(repo, threshold):
    issues = repo.get_issues(state='closed', sort='updated', direction='desc')
    print('Here are totally {} closed issues in the repository'.format(issues.totalCount))
    issue_solution_pairs = []
    count = 0

    for issue in tqdm(issues):
        if count >= threshold:
            break
        issue_title = issue.title
        issue_body = issue.body
        solutions = []
        comments = issue.get_comments()
        for comment in comments:
            if 'solution' in comment.body.lower():
                reactions = comment.reactions.get('+1', 0)
                solutions.append((comment.body, reactions))
        if solutions:
            solutions = sorted(solutions, key=lambda x: x[1], reverse=True)
            #best_solution = solutions[0][0]
            top_k = 3
            best_solutions = [sol[0] for sol in solutions[:top_k]]
            best_solutions = '\n'.join([f'Rank {i+1}: {solution}' for i, solution in enumerate(best_solutions)])
        else:
            #best_solutions = [None, None, None]
            best_solutions = "No solutions"
        pair = {
            'issue_title': issue_title,
            'issue_body': issue_body,
            'solution': best_solutions,
            'count': count
        }
        issue_solution_pairs.append(pair)
        count += 1

    return issue_solution_pairs

def main():
    parser = argparse.ArgumentParser(description='Download closed issues and comments from a GitHub repository.')
    parser.add_argument('--LIB', type=str, required=True, help='Library name')
    #parser.add_argument('--repo', type=str, required=True, help='GitHub repository in the format owner/repo')
    parser.add_argument('--token', type=str, required=True, help='GitHub API token')
    parser.add_argument('--threshold', type=int, default=3000, help='Number of issues to fetch')
    args = parser.parse_args()

    LIB = args.LIB
    GITHUB_API_TOKEN = args.token
    #repo_name = args.repo
    info_json = get_all_variable_from_cheatsheet(args.LIB)
    GITHUB_LINK = info_json['GITHUB_LINK']
    repo_name = GITHUB_LINK.replace('https://github.com/','').replace('.git','')
    if repo_name.endswith('/'):
        repo_name = repo_name[:-1]
    threshold = args.threshold

    OUTPUT_DIR = f'data/github_issues/{LIB}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    g = Github(GITHUB_API_TOKEN)
    repo = g.get_repo(repo_name)

    issue_solution_pairs = fetch_issues(repo, threshold)

    with open(os.path.join(OUTPUT_DIR, 'github_issues.json'), 'w') as f:
        json.dump(issue_solution_pairs, f, indent=4)

if __name__ == "__main__":
    main()
