#!/usr/bin/env python3
"""
GitHub Issue Creation Helper Script

This script helps create GitHub issues from the generated issue data.
You can use this with the GitHub CLI or GitHub API to batch create issues.

Usage:
    # Using GitHub CLI (requires 'gh' command)
    python3 create_github_issues.py --method=cli
    
    # Generate individual gh commands to run manually
    python3 create_github_issues.py --method=commands
    
    # Display issues for manual creation
    python3 create_github_issues.py --method=display
"""

import json
import argparse
from pathlib import Path

def load_issues():
    """Load issues from the generated JSON file."""
    json_path = Path(__file__).parent / "github_issues.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_labels_commands(labels):
    """Generate gh commands to create labels."""
    commands = []
    for label in labels:
        cmd = f"gh label create '{label['name']}' --description '{label['description']}' --color '{label['color']}'"
        commands.append(cmd)
    return commands

def create_milestones_commands(milestones):
    """Generate gh commands to create milestones."""
    commands = []
    for milestone in milestones:
        cmd = f"gh milestone create '{milestone['title']}' --description '{milestone['description']}' --due-date '{milestone['due_on']}'"
        commands.append(cmd)
    return commands

def create_issues_commands(issues):
    """Generate gh commands to create issues."""
    commands = []
    for i, issue in enumerate(issues, 1):
        # Prepare labels
        labels_str = ','.join(issue['labels'])
        
        # Prepare milestone
        milestone_str = f"--milestone '{issue['milestone']}'" if 'milestone' in issue else ""
        
        # Escape quotes and prepare body
        body = issue['body'].replace("'", "'\"'\"'")
        title = issue['title'].replace("'", "'\"'\"'")
        
        cmd = f"gh issue create --title '{title}' --body '{body}' --label '{labels_str}' {milestone_str}"
        commands.append(f"# Issue #{i}: {issue['title']}")
        commands.append(cmd)
        commands.append("")  # Empty line for readability
    
    return commands

def display_issues_summary(data):
    """Display a summary of issues for manual creation."""
    print(f"\n📋 GitHub Issues Summary")
    print(f"{'=' * 50}")
    print(f"Total Issues: {len(data['issues'])}")
    print(f"Labels: {len(data['labels'])}")
    print(f"Milestones: {len(data['milestones'])}")
    print()
    
    print("🏷️  Labels to Create:")
    for label in data['labels']:
        print(f"  • {label['name']} ({label['color']}) - {label['description']}")
    print()
    
    print("🎯 Milestones to Create:")
    for milestone in data['milestones']:
        print(f"  • {milestone['title']} (due: {milestone['due_on']}) - {milestone['description']}")
    print()
    
    print("📋 Issues by Phase:")
    phase_counts = {}
    for issue in data['issues']:
        phase_labels = [label for label in issue['labels'] if label.startswith('phase-')]
        if phase_labels:
            phase = phase_labels[0]
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        else:
            phase_counts['backlog'] = phase_counts.get('backlog', 0) + 1
    
    for phase, count in sorted(phase_counts.items()):
        print(f"  • {phase}: {count} issues")
    print()

def main():
    parser = argparse.ArgumentParser(description="Create GitHub issues from generated data")
    parser.add_argument('--method', choices=['cli', 'commands', 'display'], default='display',
                       help='Method to create issues')
    
    args = parser.parse_args()
    
    try:
        data = load_issues()
    except FileNotFoundError:
        print("❌ Error: github_issues.json not found. Run github_issues_generator.py first.")
        return
    
    if args.method == 'display':
        display_issues_summary(data)
        print("💡 Next Steps:")
        print("1. Create labels and milestones in your GitHub repository")
        print("2. Use --method=commands to generate GitHub CLI commands")
        print("3. Or manually create issues using DETAILED_ISSUES.md")
        
    elif args.method == 'commands':
        print("# GitHub Labels Creation Commands")
        print("# Run these commands in your repository directory\n")
        
        label_commands = create_labels_commands(data['labels'])
        for cmd in label_commands:
            print(cmd)
        
        print("\n# GitHub Milestones Creation Commands\n")
        milestone_commands = create_milestones_commands(data['milestones'])
        for cmd in milestone_commands:
            print(cmd)
        
        print("\n# GitHub Issues Creation Commands\n")
        issue_commands = create_issues_commands(data['issues'])
        for cmd in issue_commands:
            print(cmd)
    
    elif args.method == 'cli':
        print("🚀 Creating GitHub issues with GitHub CLI...")
        # Note: This would require actual GitHub CLI execution
        # For now, just display the commands
        print("❌ CLI execution not implemented yet.")
        print("💡 Use --method=commands to get commands to run manually.")

if __name__ == "__main__":
    main()