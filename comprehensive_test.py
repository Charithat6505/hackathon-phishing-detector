"""
Phase 6: Comprehensive System Testing
Tests the entire phishing detection pipeline
"""

import requests
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import box

console = Console()

API_URL = "http://localhost:8000"

# Comprehensive test cases covering all scenarios
TEST_CASES = [
    # Category 1: Legitimate Sites (Should be SAFE/LOW)
    {
        "url": "https://www.google.com",
        "category": "Legitimate - Trusted",
        "expected_risk": ["SAFE", "LOW"],
        "description": "Major search engine (whitelisted)"
    },
    {
        "url": "https://github.com",
        "category": "Legitimate - Trusted",
        "expected_risk": ["SAFE", "LOW"],
        "description": "Developer platform (whitelisted)"
    },
    {
        "url": "https://www.wikipedia.org",
        "category": "Legitimate - Trusted",
        "expected_risk": ["SAFE", "LOW"],
        "description": "Educational site (whitelisted)"
    },
    {
        "url": "https://www.netflix.com",
        "category": "Legitimate - Trusted",
        "expected_risk": ["SAFE", "LOW"],
        "description": "Streaming service (whitelisted)"
    },
    {
        "url": "https://stackoverflow.com",
        "category": "Legitimate - Trusted",
        "expected_risk": ["SAFE", "LOW"],
        "description": "Q&A platform (whitelisted)"
    },
    
    # Category 2: Legitimate but not whitelisted
    {
        "url": "https://www.geeksforgeeks.org",
        "category": "Legitimate - Unknown",
        "expected_risk": ["SAFE", "LOW"],
        "description": "Educational site (ML prediction)"
    },
    {
        "url": "https://www.w3schools.com",
        "category": "Legitimate - Unknown",
        "expected_risk": ["SAFE", "LOW", "MEDIUM"],
        "description": "Tutorial site (ML prediction)"
    },
    
    # Category 3: Suspicious Patterns
    {
        "url": "http://secure-paypal-verify.xyz/login",
        "category": "Phishing - Fake Domain",
        "expected_risk": ["HIGH", "CRITICAL"],
        "description": "Fake PayPal with suspicious TLD"
    },
    {
        "url": "http://account-verification-required.com/signin",
        "category": "Phishing - Suspicious Keywords",
        "expected_risk": ["MEDIUM", "HIGH", "CRITICAL"],
        "description": "Account verification scam"
    },
    {
        "url": "http://update-your-account-now.xyz",
        "category": "Phishing - Urgency Tactic",
        "expected_risk": ["MEDIUM", "HIGH"],
        "description": "Urgency-based phishing"
    },
    
    # Category 4: IP Address URLs
    {
        "url": "http://192.168.1.1/admin/login.php",
        "category": "Phishing - IP Address",
        "expected_risk": ["HIGH", "CRITICAL"],
        "description": "IP-based URL (suspicious)"
    },
    {
        "url": "http://10.0.0.1/secure/banking",
        "category": "Phishing - IP Address",
        "expected_risk": ["HIGH", "CRITICAL"],
        "description": "Private IP with banking path"
    },
    
    # Category 5: Non-existent Domains
    {
        "url": "https://this-site-does-not-exist-12345.void",
        "category": "Phishing - Fake Domain",
        "expected_risk": ["HIGH", "CRITICAL"],
        "description": "Domain doesn't exist (DNS fail)"
    },
    {
        "url": "http://absolutely-fake-domain-xyz.invalid",
        "category": "Phishing - Fake Domain",
        "expected_risk": ["HIGH", "CRITICAL"],
        "description": "Invalid TLD"
    },
    
    # Category 6: URL Shorteners
    {
        "url": "https://bit.ly/test123",
        "category": "URL Shortener",
        "expected_risk": ["SAFE", "LOW", "MEDIUM"],
        "description": "Bit.ly shortener (neutral)"
    },
    
    # Category 7: Complex Phishing
    {
        "url": "http://secure-login-paypal-verify-account-update.xyz/auth/signin.php",
        "category": "Phishing - Complex",
        "expected_risk": ["HIGH", "CRITICAL"],
        "description": "Multiple suspicious keywords"
    },
    {
        "url": "https://www.paypal-secure-login.com.fake-site.xyz",
        "category": "Phishing - Brand Spoofing",
        "expected_risk": ["MEDIUM", "HIGH", "CRITICAL"],
        "description": "PayPal brand spoofing"
    },
]

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def analyze_url(url):
    """Analyze a single URL"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"url": url},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)[:50]

def run_comprehensive_test():
    """Run all test cases"""
    
    console.print(Panel.fit(
        "[bold cyan]🔒 Phishing Detector - Comprehensive System Test[/bold cyan]\n"
        f"Testing {len(TEST_CASES)} scenarios",
        border_style="cyan"
    ))
    
    # Check API
    if not check_api_health():
        console.print("[red bold]❌ API is not running![/red bold]")
        console.print("[yellow]Start it with: python backend_api.py[/yellow]")
        return
    
    console.print("[green]✅ API is healthy[/green]\n")
    
    # Run tests
    results = []
    passed = 0
    failed = 0
    
    console.print("[cyan]Running tests...[/cyan]\n")
    
    for test_case in track(TEST_CASES, description="Testing URLs"):
        url = test_case["url"]
        expected = test_case["expected_risk"]
        
        # Analyze
        data, error = analyze_url(url)
        
        if error:
            results.append({
                "url": url,
                "category": test_case["category"],
                "expected": expected,
                "actual": f"ERROR: {error}",
                "passed": False,
                "data": None
            })
            failed += 1
        else:
            actual_risk = data["risk_level"]
            is_correct = actual_risk in expected
            
            results.append({
                "url": url,
                "category": test_case["category"],
                "expected": expected,
                "actual": actual_risk,
                "passed": is_correct,
                "data": data
            })
            
            if is_correct:
                passed += 1
            else:
                failed += 1
        
        time.sleep(0.5)  # Rate limiting
    
    # Display results
    console.print("\n" + "="*80 + "\n")
    console.print("[bold cyan]📊 Test Results Summary[/bold cyan]\n")
    
    # Summary table
    summary_table = Table(title="Test Summary", box=box.ROUNDED)
    summary_table.add_column("Category", style="cyan", width=25)
    summary_table.add_column("URL", style="white", width=35)
    summary_table.add_column("Expected", style="yellow", width=15)
    summary_table.add_column("Actual", style="white", width=15)
    summary_table.add_column("Status", style="bold")
    
    # Group by category
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    # Display by category
    for category, tests in categories.items():
        for i, test in enumerate(tests):
            url_short = test["url"][:35] + "..." if len(test["url"]) > 35 else test["url"]
            expected_str = "/".join(test["expected"]) if isinstance(test["expected"], list) else test["expected"]
            
            if test["passed"]:
                status = "[green]✅ PASS[/green]"
                actual_color = "[green]" + test["actual"] + "[/green]"
            else:
                status = "[red]❌ FAIL[/red]"
                actual_color = "[red]" + test["actual"] + "[/red]"
            
            cat_display = category if i == 0 else ""
            
            summary_table.add_row(
                cat_display,
                url_short,
                expected_str,
                actual_color,
                status
            )
        
        summary_table.add_row("", "", "", "", "")  # Spacer
    
    console.print(summary_table)
    
    # Statistics
    total = len(results)
    pass_rate = (passed / total) * 100
    
    stats_table = Table(box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan bold")
    stats_table.add_column("Value", style="white bold")
    
    stats_table.add_row("Total Tests", str(total))
    stats_table.add_row("Passed", f"[green]{passed}[/green]")
    stats_table.add_row("Failed", f"[red]{failed}[/red]")
    stats_table.add_row("Pass Rate", f"[{'green' if pass_rate >= 80 else 'yellow'}]{pass_rate:.1f}%[/]")
    
    console.print("\n")
    console.print(stats_table)
    
    # Detailed failures
    if failed > 0:
        console.print("\n[yellow]⚠️ Failed Test Details:[/yellow]\n")
        
        for result in results:
            if not result["passed"]:
                console.print(f"[red]❌ {result['category']}[/red]")
                console.print(f"   URL: {result['url']}")
                console.print(f"   Expected: {result['expected']}")
                console.print(f"   Got: {result['actual']}")
                
                if result["data"]:
                    console.print(f"   Phishing Probability: {result['data']['features']['phishing_probability']:.1f}%")
                
                console.print()
    
    # Final verdict
    console.print("\n" + "="*80 + "\n")
    
    if pass_rate >= 90:
        console.print("[green bold]🎉 EXCELLENT! System is working great![/green bold]")
    elif pass_rate >= 75:
        console.print("[yellow bold]✅ GOOD! System is working well with minor issues.[/yellow bold]")
    else:
        console.print("[red bold]⚠️ NEEDS IMPROVEMENT! Review failed cases.[/red bold]")
    
    # Performance metrics
    if results:
        avg_time = sum(r["data"]["analysis_time"] for r in results if r["data"]) / len([r for r in results if r["data"]])
        console.print(f"\n[cyan]Average Analysis Time: {avg_time:.2f}s[/cyan]")
    
    return results

def generate_report(results):
    """Generate a detailed test report"""
    
    console.print("\n" + "="*80)
    console.print("[bold cyan]📄 Generating Detailed Report...[/bold cyan]")
    
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("PHISHING DETECTOR - COMPREHENSIVE TEST REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        failed = total - passed
        pass_rate = (passed / total) * 100
        
        f.write(f"SUMMARY\n")
        f.write(f"-------\n")
        f.write(f"Total Tests: {total}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Pass Rate: {pass_rate:.1f}%\n\n")
        
        # Detailed results
        f.write("DETAILED RESULTS\n")
        f.write("----------------\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Test {i}: {result['category']}\n")
            f.write(f"URL: {result['url']}\n")
            f.write(f"Expected Risk: {result['expected']}\n")
            f.write(f"Actual Risk: {result['actual']}\n")
            f.write(f"Status: {'PASS' if result['passed'] else 'FAIL'}\n")
            
            if result['data']:
                f.write(f"Phishing Probability: {result['data']['features']['phishing_probability']:.2f}%\n")
                f.write(f"Confidence: {result['data']['confidence']*100:.2f}%\n")
                f.write(f"Analysis Time: {result['data']['analysis_time']:.3f}s\n")
            
            f.write("\n")
    
    console.print(f"[green]✅ Report saved to: {report_file}[/green]")

if __name__ == "__main__":
    console.print("\n")
    results = run_comprehensive_test()
    
    if results:
        console.print("\n")
        save_report = console.input("[cyan]Save detailed report to file? (y/n):[/cyan] ")
        
        if save_report.lower() == 'y':
            generate_report(results)
    
    console.print("\n[bold]Testing complete![/bold]\n")



# ╭──────────────────────────────────────────────────╮
# │ 🔒 Phishing Detector - Comprehensive System Test │
# │ Testing 17 scenarios                             │
# ╰──────────────────────────────────────────────────╯
# ✅ API is healthy

# Running tests...

# Testing URLs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:29

# ================================================================================

# 📊 Test Results Summary

#                                                   Test Summary                                                   
# ╭───────────────────────────┬─────────────────────────────────────┬─────────────────┬─────────────────┬─────────╮
# │ Category                  │ URL                                 │ Expected        │ Actual          │ Status  │
# ├───────────────────────────┼─────────────────────────────────────┼─────────────────┼─────────────────┼─────────┤
# │ Legitimate - Trusted      │ https://www.google.com              │ SAFE/LOW        │ SAFE            │ ✅ PASS │
# │                           │ https://github.com                  │ SAFE/LOW        │ SAFE            │ ✅ PASS │
# │                           │ https://www.wikipedia.org           │ SAFE/LOW        │ SAFE            │ ✅ PASS │
# │                           │ https://www.netflix.com             │ SAFE/LOW        │ SAFE            │ ✅ PASS │
# │                           │ https://stackoverflow.com           │ SAFE/LOW        │ SAFE            │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# │ Legitimate - Unknown      │ https://www.geeksforgeeks.org       │ SAFE/LOW        │ SAFE            │ ✅ PASS │
# │                           │ https://www.w3schools.com           │ SAFE/LOW/MEDIUM │ LOW             │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# │ Phishing - Fake Domain    │ http://secure-paypal-verify.xyz/lo… │ HIGH/CRITICAL   │ HIGH            │ ✅ PASS │
# │                           │ https://this-site-does-not-exist-1… │ HIGH/CRITICAL   │ HIGH            │ ✅ PASS │
# │                           │ http://absolutely-fake-domain-xyz.… │ HIGH/CRITICAL   │ HIGH            │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# │ Phishing - Suspicious     │ http://account-verification-requir… │ MEDIUM/HIGH/CR… │ HIGH            │ ✅ PASS │
# │ Keywords                  │                                     │                 │                 │         │
# │                           │                                     │                 │                 │         │
# │ Phishing - Urgency Tactic │ http://update-your-account-now.xyz  │ MEDIUM/HIGH     │ HIGH            │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# │ Phishing - IP Address     │ http://192.168.1.1/admin/login.php  │ HIGH/CRITICAL   │ CRITICAL        │ ✅ PASS │
# │                           │ http://10.0.0.1/secure/banking      │ HIGH/CRITICAL   │ CRITICAL        │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# │ URL Shortener             │ https://bit.ly/test123              │ SAFE/LOW/MEDIUM │ MEDIUM          │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# │ Phishing - Complex        │ http://secure-login-paypal-verify-… │ HIGH/CRITICAL   │ HIGH            │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# │ Phishing - Brand Spoofing │ https://www.paypal-secure-login.co… │ MEDIUM/HIGH/CR… │ HIGH            │ ✅ PASS │
# │                           │                                     │                 │                 │         │
# ╰───────────────────────────┴─────────────────────────────────────┴─────────────────┴─────────────────┴─────────╯


# ╭─────────────┬────────╮
# │ Metric      │ Value  │
# ├─────────────┼────────┤
# │ Total Tests │ 17     │
# │ Passed      │ 17     │
# │ Failed      │ 0      │
# │ Pass Rate   │ 100.0% │
# ╰─────────────┴────────╯

# ================================================================================

# 🎉 EXCELLENT! System is working great!

# Average Analysis Time: 2.70s


# Save detailed report to file? (y/n): y

# ================================================================================
# 📄 Generating Detailed Report...
# ✅ Report saved to: test_report_20251122_090059.txt

# Testing complete!