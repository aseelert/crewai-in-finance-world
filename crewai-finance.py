from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, CSVSearchTool, FileReadTool, DirectoryReadTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pathlib import Path

dir_tool_read = DirectoryReadTool(directory='company_analysis')

def download_company_data(company_ticker):
    """Download and store financial data for a company"""
    symbol = yf.Ticker(company_ticker)
    df = symbol.history(period="5y")

    # Create directory if it doesn't exist
    Path("company_analysis").mkdir(exist_ok=True)
    # Add ticker column to dataframe
    df['Ticker'] = company_ticker

    # Save historical data to CSV
    df.to_csv(f"company_analysis/{company_ticker}_data.csv")
    print(f"Saved {company_ticker} data")
    return df


# Load environment variables
load_dotenv()

def get_historical_analysis(ticker, company_name):
    """Get 5-year historical data and analyze major events impact"""
    # Get stock data
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years
    hist_data = stock.history(start=start_date, end=end_date)

    # Calculate key metrics
    hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
    hist_data['SMA_200'] = hist_data['Close'].rolling(window=200).mean()

    # Create price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name='Price'
    ))

    # Save chart
    output_dir = Path("company_analysis/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(f"company_analysis/charts/{ticker}_5year_chart.html")

    return hist_data

def analyze_major_events(company_name, ticker):
    """Analyze impact of major events on stock price"""
    major_events = {
        # Recent Political Events
        "2024 US Presidential Election": "2024-11-05",
        "US Capitol Riot": "2021-01-06",
        "Biden Inauguration": "2021-01-20",

        # Recent Economic Events
        "Fed Rate Hikes Start": "2022-03-16",
        "Silicon Valley Bank Crisis": "2023-03-10",
        "Credit Suisse Collapse": "2023-03-15",
        "Inflation Peak": "2022-06-15",

        # Recent Tech Events
        "ChatGPT Launch": "2022-11-30",
        "Meta/Facebook Rebrand": "2021-10-28",
        "Twitter/X Acquisition": "2022-10-27",
        "FTX Collapse": "2022-11-11",

        # Recent Global Events
        "Ukraine War": "2022-02-24",
        "COVID-19 Emergency End": "2023-05-05",
        "Israel-Hamas War": "2023-10-07",

        # Financial Market Events
        "S&P 500 Record High": "2024-01-19",
        "Nvidia Joins Dow": "2024-02-20",
        "Bitcoin ETF Approval": "2024-01-10"
    }

    stock = yf.Ticker(ticker)
    analysis = []

    for event, date in major_events.items():
        # Get stock data around the event
        start_date = pd.to_datetime(date) - timedelta(days=30)
        end_date = pd.to_datetime(date) + timedelta(days=30)

        event_data = stock.history(start=start_date, end=end_date)
        if not event_data.empty:
            pre_event = event_data['Close'].iloc[0]
            post_event = event_data['Close'].iloc[-1]
            change_pct = ((post_event - pre_event) / pre_event) * 100

            analysis.append({
                "event": event,
                "date": date,
                "impact_percentage": round(change_pct, 2),
                "direction": "increase" if change_pct > 0 else "decrease"
            })

    return analysis

# Create tools for accessing CSV files
#csv_tool = CSVSearchTool(directory="company_analysis")
#csv_tool = FileReadTool(file_path="company_analysis/IBM_data.csv")


# Enhanced researcher agent with CSV access
researcher = Agent(
    role='Financial Research Analyst',
    goal='Analyze companies using comprehensive historical data and event impact analysis',
    backstory="""You are an expert financial analyst specializing in historical
    market analysis and event impact assessment. You excel at identifying
    correlations between major events and stock performance.""",
    verbose=True,
    tools=[SerperDevTool(), dir_tool_read],  # Add CSV and Directory tools
    allow_delegation=False,
    context="""You have access to historical price data in CSV files located in the
    company_analysis directory. For each company (e.g., IBM, INFY), you can:

    1. Use the CSVSearchTool to analyze the data with queries like:
       - "Find the highest closing price for {company_name}"
       - "Calculate average volume for {company_name}"
       - "Compare price trends between dates during major events"

    2. Access key columns:
       - Date: Trading date
       - Adj Close: Adjusted closing price
       - Close: Closing price
       - High: Daily high
       - Low: Daily low
       - Open: Opening price
       - Volume: Trading volume
       - Ticker: Company ticker symbol

    Use the tools to analyze trends, identify patterns, and correlate with major events."""
)

# Enhanced writer agent
writer = Agent(
    role='Financial Content Strategist',
    goal='Create detailed company analysis reports with historical context',
    backstory="""You are a skilled financial writer who excels at explaining
    complex market movements and their relationships to world events. You make
    data-driven insights accessible and actionable.""",
    verbose=True
)

def create_csv_tool(ticker):
    """Create a FileReadTool for the specific company's CSV file"""
    return FileReadTool(
        file_path=f"company_analysis/{ticker}_data.csv",
        description=f"""This CSV contains historical stock market data for {ticker} with columns:
        - Date: Trading date with timezone
        - Open: Opening price for the trading day
        - High: Highest price during the trading day
        - Low: Lowest price during the trading day
        - Close: Closing price for the trading day
        - Volume: Number of shares traded
        - Dividends: Dividend payments
        - Stock Splits: Stock split events
        - Ticker: Company symbol ({ticker})

        Use this data to:
        1. Analyze price trends and patterns
        2. Calculate daily returns
        3. Identify significant volume changes
        4. Track corporate actions (splits/dividends)
        5. Compare price levels during major events"""
    )

def create_csv_context(ticker):
    """Create context explaining CSV data structure and the actual data for the agent"""
    # Read CSV file
    df = pd.read_csv(f"company_analysis/{ticker}_data.csv")

    context = f"""I am providing you with historical stock market data for {ticker} from a CSV file.
    Each row represents one trading day with the following columns:

    1. Date: Trading date with timezone offset in format 'YYYY-MM-DD HH:MM:SS-TZ:00'
       Example: 2019-11-14 00:00:00-05:00
       - This represents the exact trading day in Eastern Time (ET)

    2. Open: The first trading price of the stock for that day in USD
       Example: 101.62050219343985
       - This is the price at market open (9:30 AM ET)

    3. High: The highest trading price reached during that day in USD
       Example: 101.77203019967824
       - This represents the peak price for the trading day

    4. Low: The lowest trading price reached during that day in USD
       Example: 101.014402260738
       - This represents the bottom price for the trading day

    5. Close: The final trading price for that day in USD
       Example: 101.52201080322266
       - This is the last price when market closes (4:00 PM ET)

    6. Volume: Total number of shares traded that day
       Example: 4425940
       - Higher volume often indicates more significant price movements

    7. Dividends: Any dividend payments made that day (0.0 if none)
       Example: 0.0
       - Cash payments distributed to shareholders

    8. Stock Splits: Any stock splits that occurred (0.0 if none)
       Example: 0.0
       - When a company multiplies its shares and adjusts price proportionally

    9. Ticker: The stock symbol ({ticker})
       - Unique identifier for the company on stock exchanges

    Here is the actual data for analysis:

    {df}

    The data is ordered chronologically from oldest to newest dates.
    All price values are in US dollars.
    This data represents actual historical trading information and should be used as the source of truth for any analysis.

    When analyzing trends, pay special attention to:
    - The relationship between Open and Close prices (indicates daily trend)
    - The High-Low range (indicates daily volatility)
    - Volume changes (indicates trading activity/interest)
    - Any dividends or stock splits that might affect price comparisons"""
    print(context)
    return context

# Create researcher agent with CSV context
def create_research_task(company_name, ticker):
    # Create CSV context for this company
    csv_context = create_csv_context(ticker)

    researcher = Agent(
        role='Financial Research Analyst',
        goal='Analyze companies using comprehensive historical data and event impact analysis',
        backstory="""You are an expert financial analyst specializing in historical
        market analysis and event impact assessment. You excel at identifying
        correlations between major events and stock performance.""",
        verbose=True,
        tools=[SerperDevTool()],
        allow_delegation=False,
        context=csv_context
    )

    return Task(
        description=f"""Research and analyze {company_name} ({ticker}):
        1. Analyze historical price trends and patterns
        2. Identify significant price movements
        3. Compare performance during major market events
        4. Assess current market position""",
        expected_output="""A detailed analysis report including:
        1. Historical price trends with specific values and dates
        2. Major price movements and their causes
        3. Performance comparison during key market events
        4. Current market position assessment
        5. Key statistics and metrics""",
        agent=researcher
    )

def create_writing_task(company_name, ticker, hist_analysis):
    return Task(
        description=f"""Create comprehensive report for {company_name} ({ticker}):

        # {company_name} Historical Analysis Report

        ## Current Market Position
        - Latest stock price: $[PRICE]
        - Market Cap: $[MARKET_CAP]
        - P/E Ratio: [PE_RATIO]
        - 52-week Range: $[52W_LOW] - $[52W_HIGH]
        - YTD Performance: [YTD_PERF]%

        ## Historical Performance (5-Year Analysis)
        ### Price Extremes
        | Metric | Price | Date |
        |--------|-------|------|
        | All-Time High | $[ATH_PRICE] | [ATH_DATE] |
        | All-Time Low | $[ATL_PRICE] | [ATL_DATE] |
        | 5-Year High | $[5Y_HIGH] | [5Y_HIGH_DATE] |
        | 5-Year Low | $[5Y_LOW] | [5Y_LOW_DATE] |

        ### Top 10 Daily Gains
        | Date | Price Change | % Change | Closing Price |
        |------|--------------|----------|---------------|
        [TOP_10_GAINS_TABLE]

        ### Top 10 Daily Losses
        | Date | Price Change | % Change | Closing Price |
        |------|--------------|----------|---------------|
        [TOP_10_LOSSES_TABLE]

        ### Key Trends and Patterns
        - Major support levels: $[SUPPORT_LEVELS]
        - Major resistance levels: $[RESISTANCE_LEVELS]
        - Moving averages analysis
        - Volume trends
        - Seasonality patterns

        ## Event Impact Analysis
        ### Major Market Events Impact
        | Event | Date | Impact (%) | Direction |
        |-------|------|------------|-----------|
        {hist_analysis}

        ## Risk Assessment
        ### Historical Volatility Analysis
        - Beta coefficient: [BETA]
        - Standard deviation: [STD_DEV]
        - Sharpe ratio: [SHARPE]

        ### Event Sensitivity
        - Correlation with market indices
        - Response to economic indicators
        - Sector-specific risks

        ### Market Correlation
        - Correlation with S&P 500: [SP500_CORR]
        - Correlation with sector ETF: [SECTOR_CORR]
        - Correlation with competitors

        ## Future Outlook
        ### Technical Indicators
        - RSI: [RSI]
        - MACD: [MACD]
        - Moving averages trends

        ### Market Sentiment
        - Analyst recommendations
        - Institutional holdings
        - Short interest

        ### Growth Potential
        - Revenue growth projections
        - Market expansion opportunities
        - Innovation pipeline""",
        expected_output="A detailed markdown report with comprehensive historical analysis, price extremes, top movers, and event correlations",
        agent=writer
    )

# Process companies
companies = [
    {"name": "IBM", "ticker": "IBM"},
    {"name": "Infosys", "ticker": "INFY"}
]

for company in companies:
    download_company_data(company["ticker"])
    # Get historical data and event analysis
    hist_data = get_historical_analysis(company["ticker"], company["name"])
    event_analysis = analyze_major_events(company["name"], company["ticker"])

    # Create company-specific tools and tasks
    company_csv_tool = create_csv_tool(company["ticker"])

    # Create researcher agent with company-specific context
    researcher = Agent(
        role='Financial Research Analyst',
        goal='Analyze companies using comprehensive historical data and event impact analysis',
        backstory="""You are an expert financial analyst specializing in historical
        market analysis and event impact assessment. You excel at identifying
        correlations between major events and stock performance.""",
        verbose=True,
        tools=[SerperDevTool(), company_csv_tool],  # Add company-specific CSV tool
        allow_delegation=False,
        context=f"""You are analyzing {company['name']} ({company['ticker']}) using historical market data.
        The CSV file contains daily trading data from 2020 to present.

        Key analysis points:
        1. Price Trends: Look for patterns in closing prices
        2. Volume Analysis: Identify unusual trading activity
        3. Event Impact: Compare price changes during major events
        4. Technical Indicators: Calculate moving averages and support levels
        5. Risk Metrics: Analyze volatility and drawdowns

        Use the FileReadTool to access the CSV data at: company_analysis/{company['ticker']}_data.csv"""
    )

    # Create tasks with company-specific tools
    research_task = create_research_task(company["name"], company["ticker"])
    writing_task = create_writing_task(company["name"], company["ticker"], event_analysis)

    # Create and run crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
        process=Process.sequential
    )

    # Get and save results
    result = crew.kickoff()

    # Save report with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("company_analysis")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f"{company['name']}_analysis_{timestamp}.md", "w") as f:
        f.write(str(result.tasks_output[1]))

