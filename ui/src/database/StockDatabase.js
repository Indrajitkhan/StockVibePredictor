// Comprehensive stock database for auto-suggestion functionality
// This includes major stocks from various exchanges with company names and ticker symbols

export const stockDatabase = [
  // Technology - FAANG & Major Tech
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "GOOGL", name: "Alphabet Inc. Class A" },
  { symbol: "GOOG", name: "Alphabet Inc. Class C" },
  { symbol: "MSFT", name: "Microsoft Corporation" },
  { symbol: "AMZN", name: "Amazon.com Inc." },
  { symbol: "META", name: "Meta Platforms Inc." },
  { symbol: "TSLA", name: "Tesla Inc." },
  { symbol: "NFLX", name: "Netflix Inc." },
  { symbol: "NVDA", name: "NVIDIA Corporation" },
  { symbol: "CRM", name: "Salesforce Inc." },
  { symbol: "ORCL", name: "Oracle Corporation" },
  { symbol: "ADBE", name: "Adobe Inc." },
  { symbol: "INTC", name: "Intel Corporation" },
  { symbol: "AMD", name: "Advanced Micro Devices Inc." },
  { symbol: "IBM", name: "International Business Machines Corporation" },
  { symbol: "CSCO", name: "Cisco Systems Inc." },
  { symbol: "AVGO", name: "Broadcom Inc." },
  { symbol: "TXN", name: "Texas Instruments Incorporated" },
  { symbol: "QCOM", name: "QUALCOMM Incorporated" },
  { symbol: "AMAT", name: "Applied Materials Inc." },
  { symbol: "MU", name: "Micron Technology Inc." },
  { symbol: "LRCX", name: "Lam Research Corporation" },
  { symbol: "ADI", name: "Analog Devices Inc." },
  { symbol: "KLAC", name: "KLA Corporation" },
  { symbol: "MRVL", name: "Marvell Technology Inc." },

  // Electric Vehicles & Transportation
  { symbol: "RIVN", name: "Rivian Automotive Inc." },
  { symbol: "LCID", name: "Lucid Group Inc." },
  { symbol: "NIO", name: "NIO Inc." },
  { symbol: "XPEV", name: "XPeng Inc." },
  { symbol: "LI", name: "Li Auto Inc." },
  { symbol: "F", name: "Ford Motor Company" },
  { symbol: "GM", name: "General Motors Company" },

  // Financial Services
  { symbol: "JPM", name: "JPMorgan Chase & Co." },
  { symbol: "BAC", name: "Bank of America Corporation" },
  { symbol: "WFC", name: "Wells Fargo & Company" },
  { symbol: "GS", name: "The Goldman Sachs Group Inc." },
  { symbol: "MS", name: "Morgan Stanley" },
  { symbol: "C", name: "Citigroup Inc." },
  { symbol: "V", name: "Visa Inc." },
  { symbol: "MA", name: "Mastercard Incorporated" },
  { symbol: "PYPL", name: "PayPal Holdings Inc." },
  { symbol: "AXP", name: "American Express Company" },
  { symbol: "BRK.A", name: "Berkshire Hathaway Inc. Class A" },
  { symbol: "BRK.B", name: "Berkshire Hathaway Inc. Class B" },

  // Healthcare & Biotech
  { symbol: "JNJ", name: "Johnson & Johnson" },
  { symbol: "PFE", name: "Pfizer Inc." },
  { symbol: "UNH", name: "UnitedHealth Group Incorporated" },
  { symbol: "MRNA", name: "Moderna Inc." },
  { symbol: "BNTX", name: "BioNTech SE" },
  { symbol: "ABBV", name: "AbbVie Inc." },
  { symbol: "TMO", name: "Thermo Fisher Scientific Inc." },
  { symbol: "ABT", name: "Abbott Laboratories" },
  { symbol: "DHR", name: "Danaher Corporation" },
  { symbol: "BMY", name: "Bristol-Myers Squibb Company" },
  { symbol: "LLY", name: "Eli Lilly and Company" },
  { symbol: "MRK", name: "Merck & Co. Inc." },

  // Entertainment & Media
  { symbol: "DIS", name: "The Walt Disney Company" },
  { symbol: "CMCSA", name: "Comcast Corporation" },
  { symbol: "T", name: "AT&T Inc." },
  { symbol: "VZ", name: "Verizon Communications Inc." },
  { symbol: "NFLX", name: "Netflix Inc." },
  { symbol: "SPOT", name: "Spotify Technology S.A." },

  // Retail & E-commerce
  { symbol: "WMT", name: "Walmart Inc." },
  { symbol: "HD", name: "The Home Depot Inc." },
  { symbol: "COST", name: "Costco Wholesale Corporation" },
  { symbol: "TGT", name: "Target Corporation" },
  { symbol: "LOW", name: "Lowe's Companies Inc." },
  { symbol: "EBAY", name: "eBay Inc." },

  // Aerospace & Defense
  { symbol: "BA", name: "The Boeing Company" },
  { symbol: "LMT", name: "Lockheed Martin Corporation" },
  { symbol: "RTX", name: "Raytheon Technologies Corporation" },
  { symbol: "NOC", name: "Northrop Grumman Corporation" },
  { symbol: "GD", name: "General Dynamics Corporation" },

  // Energy
  { symbol: "XOM", name: "Exxon Mobil Corporation" },
  { symbol: "CVX", name: "Chevron Corporation" },
  { symbol: "COP", name: "ConocoPhillips" },
  { symbol: "SLB", name: "Schlumberger Limited" },
  { symbol: "EOG", name: "EOG Resources Inc." },

  // Industrial & Manufacturing
  { symbol: "CAT", name: "Caterpillar Inc." },
  { symbol: "DE", name: "Deere & Company" },
  { symbol: "GE", name: "General Electric Company" },
  { symbol: "MMM", name: "3M Company" },
  { symbol: "HON", name: "Honeywell International Inc." },

  // Consumer Goods
  { symbol: "KO", name: "The Coca-Cola Company" },
  { symbol: "PEP", name: "PepsiCo Inc." },
  { symbol: "PG", name: "The Procter & Gamble Company" },
  { symbol: "UL", name: "Unilever PLC" },
  { symbol: "NKE", name: "NIKE Inc." },
  { symbol: "SBUX", name: "Starbucks Corporation" },

  // Cryptocurrency & Fintech
  { symbol: "COIN", name: "Coinbase Global Inc." },
  { symbol: "SQ", name: "Block Inc." },
  { symbol: "HOOD", name: "Robinhood Markets Inc." },

  // Real Estate
  { symbol: "AMT", name: "American Tower Corporation" },
  { symbol: "PLD", name: "Prologis Inc." },
  { symbol: "CCI", name: "Crown Castle International Corp." },

  // Utilities
  { symbol: "NEE", name: "NextEra Energy Inc." },
  { symbol: "SO", name: "The Southern Company" },
  { symbol: "D", name: "Dominion Energy Inc." },

  // Gaming
  { symbol: "ATVI", name: "Activision Blizzard Inc." },
  { symbol: "EA", name: "Electronic Arts Inc." },
  { symbol: "RBLX", name: "Roblox Corporation" },
  { symbol: "TTWO", name: "Take-Two Interactive Software Inc." },

  // Cloud & SaaS
  { symbol: "SNOW", name: "Snowflake Inc." },
  { symbol: "PLTR", name: "Palantir Technologies Inc." },
  { symbol: "ZM", name: "Zoom Video Communications Inc." },
  { symbol: "DOCU", name: "DocuSign Inc." },
  { symbol: "TWLO", name: "Twilio Inc." },
  { symbol: "OKTA", name: "Okta Inc." },
  { symbol: "WORK", name: "Slack Technologies Inc." },

  // Emerging Tech
  { symbol: "PLTR", name: "Palantir Technologies Inc." },
  { symbol: "ARKK", name: "ARK Innovation ETF" },
  { symbol: "QQQ", name: "Invesco QQQ Trust" },
  { symbol: "SPY", name: "SPDR S&P 500 ETF Trust" },
  { symbol: "IWM", name: "iShares Russell 2000 ETF" },
  { symbol: "VTI", name: "Vanguard Total Stock Market ETF" },

  // International Stocks (ADRs)
  { symbol: "BABA", name: "Alibaba Group Holding Limited" },
  { symbol: "JD", name: "JD.com Inc." },
  { symbol: "PDD", name: "PDD Holdings Inc." },
  { symbol: "TME", name: "Tencent Music Entertainment Group" },
  { symbol: "BIDU", name: "Baidu Inc." },
  { symbol: "TSM", name: "Taiwan Semiconductor Manufacturing Company Limited" },
  { symbol: "ASML", name: "ASML Holding N.V." },
  { symbol: "SAP", name: "SAP SE" },
  { symbol: "TM", name: "Toyota Motor Corporation" },
  { symbol: "SONY", name: "Sony Group Corporation" },

  // Indian Market - Indices and Major Stocks
  { symbol: "NIFTY", name: "NIFTY 50 Index" },
  { symbol: "NIFTY50", name: "NIFTY 50 Index" },
  { symbol: "^NSEI", name: "NIFTY 50 Index" },
  { symbol: "SENSEX", name: "BSE SENSEX" },
  { symbol: "^BSESN", name: "BSE SENSEX" },
  { symbol: "RELIANCE.NS", name: "Reliance Industries Limited" },
  { symbol: "TCS.NS", name: "Tata Consultancy Services Limited" },
  { symbol: "HDFCBANK.NS", name: "HDFC Bank Limited" },
  { symbol: "INFY.NS", name: "Infosys Limited" },
  { symbol: "HINDUNILVR.NS", name: "Hindustan Unilever Limited" },
  { symbol: "ITC.NS", name: "ITC Limited" },
  { symbol: "SBIN.NS", name: "State Bank of India" },
  { symbol: "BHARTIARTL.NS", name: "Bharti Airtel Limited" },
  { symbol: "KOTAKBANK.NS", name: "Kotak Mahindra Bank Limited" },
  { symbol: "LT.NS", name: "Larsen & Toubro Limited" },

  // Meme Stocks & Popular Retail
  { symbol: "GME", name: "GameStop Corp." },
  { symbol: "AMC", name: "AMC Entertainment Holdings Inc." },
  { symbol: "BB", name: "BlackBerry Limited" },
  { symbol: "NOK", name: "Nokia Corporation" },
  { symbol: "WISH", name: "ContextLogic Inc." },
  { symbol: "CLOV", name: "Clover Health Investments Corp." },

  // SPACs & Recent IPOs
  { symbol: "SPCE", name: "Virgin Galactic Holdings Inc." },
  { symbol: "OPEN", name: "Opendoor Technologies Inc." },
  { symbol: "SOFI", name: "SoFi Technologies Inc." },
  { symbol: "UPST", name: "Upstart Holdings Inc." },
  { symbol: "AFRM", name: "Affirm Holdings Inc." },

  // Additional Popular Stocks
  { symbol: "DKNG", name: "DraftKings Inc." },
  { symbol: "PENN", name: "PENN Entertainment Inc." },
  { symbol: "ROKU", name: "Roku Inc." },
  { symbol: "UBER", name: "Uber Technologies Inc." },
  { symbol: "LYFT", name: "Lyft Inc." },
  { symbol: "DASH", name: "DoorDash Inc." },
  { symbol: "ABNB", name: "Airbnb Inc." },
];

// Search function for auto-suggestion
export const searchStocks = (query, limit = 10) => {
  if (!query || query.length < 1) return [];

  const searchTerm = query.toLowerCase().trim();

  // Score each stock based on relevance
  const scoredResults = stockDatabase.map((stock) => {
    const symbolMatch = stock.symbol.toLowerCase().includes(searchTerm);
    const nameMatch = stock.name.toLowerCase().includes(searchTerm);
    const symbolStartsWith = stock.symbol.toLowerCase().startsWith(searchTerm);
    const nameStartsWith = stock.name.toLowerCase().startsWith(searchTerm);
    const exactSymbolMatch = stock.symbol.toLowerCase() === searchTerm;
    const exactNameMatch = stock.name.toLowerCase() === searchTerm;

    let score = 0;

    // Exact matches get highest priority
    if (exactSymbolMatch) score += 1000;
    if (exactNameMatch) score += 900;

    // Starts with matches get high priority
    if (symbolStartsWith) score += 100;
    if (nameStartsWith) score += 80;

    // Contains matches get lower priority
    if (symbolMatch) score += 50;
    if (nameMatch) score += 30;

    // Bonus for shorter symbols (easier to type)
    if (stock.symbol.length <= 4) score += 10;

    return { ...stock, score };
  });

  // Filter out stocks with no matches and sort by score
  return scoredResults
    .filter((stock) => stock.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
};

// Get popular stocks for default suggestions
export const getPopularStocks = () => [
  "AAPL",
  "GOOGL",
  "MSFT",
  "TSLA",
  "AMZN",
  "META",
  "NFLX",
  "NVDA",
  "JPM",
  "V",
  "JNJ",
  "WMT",
  "PG",
  "HD",
  "DIS",
  "PYPL",
];
