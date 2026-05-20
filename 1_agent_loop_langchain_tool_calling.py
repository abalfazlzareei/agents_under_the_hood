from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10


MODEl = "gpt-oss:20b"

@tool
def get_product_price(product:str)-> float:
    "Look up the price of a product in the catalogue"
    print(f".    >> Executing get_product_price (product={product})")
    prices = {"laptob":30, "keyboard":10, "pen":15}
    return prices.get(product,0)

@tool
def apply_discount(price:float, discount_tier:str)->float:
    """Apply a discount to the price and return the final price
        Available tiers: bronze, silver, gold. 
    """
    print(f".  >> Executing apply_discount(price={price}, discount_tier={discount_tier})")
    discount_percentages = {"bronze":5, "silver":10, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount/100),2)

# ----- Agent Loop ----
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
        tools = [get_product_price, apply_discount]
        tools_dict = {t.name: t for t in tools}
        print(f"tools_dict:{tools_dict}")

        llm = init_chat_model(f"ollama:gpt-oss:20b",temperature = 0)
        llm_with_tools = llm.bind_tools(tools)
        print("Question: ", question)
        print("="*60)
        Messages = [
              SystemMessage(
                 content=(   "You are a helpful shopping assistant"
                    "you have access to a product catalog tool"
                    "and a discount tool.\n\n"
                    "STRICT RULES - you must follow these exactly:\n"
                    "1. NEVER guess or assume any product price."
                    "You must call get_product_price first to get the real price.\n"
                    "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price"
                    "returned by get_product_price - do NOT pass a made-up number.\n"
                    "3. NEVER calculate discounts yourself using math."
                    "Always use the apply_discount tool.\n"
                    "4. If the user does not specify a discount tier,"
                    "ask them which tier to use - do NOT assume one."
                    )
              ),
              HumanMessage(content = question)
        ]

        for iteration in range(1,MAX_ITERATIONS + 1):
            print(f"\n -------Iteration {iteration} ---")
            ai_message = llm_with_tools.invoke(Messages)
            print("AI_Message:", ai_message)
            print("AI Message:", ai_message.content)
            
            tool_calls = ai_message.tool_calls
            print("Tool Calls:", ai_message.tool_calls)
            if not tool_calls:
                print(f"\n Final Answer",ai_message.content)
                return ai_message.content

            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"tool {tool_name} Not Found!")
            
            observation = tool_to_use.invoke(tool_args)
            print(f"Tool Result: {observation}")

            Messages.append(ai_message)
            Messages.append(
                 ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )

        print("ERROR: Max Iterations reached without a final answer.")
        return None

if __name__ == "__main__":
        print("Hello Langchain Agent (.bind_tools)!")
        print()
        print("what is the price of a laptob after applying a gold discount?")
        run_agent("what is the price of a laptob after applying a gold discount?")



"""
M&A Universe Coverage Analysis
================================
Answers 4 key questions about M&A deal coverage vs MSCI World universe.

DATA INPUTS:
  - df_universe         : M&A deal data  (read from prep_data/universe_data.parquet)
  - MSCI_universe_membership : boolean membership matrix, index=dates, cols=tradingItemIds
  - MSCI_world_data     : price/volume/mktcap data with cols:
                          TRADINGITEMID, PRICINGDATE, TRADINGITEM_PRIMARYFLAG,
                          MARKETCAPUSD, VOLUME_USD

Run:  python ma_universe_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#2a2d3a",
    "axes.labelcolor":  "#c9cde0",
    "axes.titlecolor":  "#e8eaf0",
    "xtick.color":      "#7a7f96",
    "ytick.color":      "#7a7f96",
    "text.color":       "#c9cde0",
    "grid.color":       "#1e2130",
    "grid.linewidth":   0.8,
    "legend.framealpha": 0.0,
    "legend.labelcolor": "#c9cde0",
    "font.family":      "monospace",
    "font.size":        10,
})

TEAL    = "#00d4b4"
AMBER   = "#f5a623"
CORAL   = "#ff6b6b"
BLUE    = "#4a9eff"
PURPLE  = "#9b7fe8"
GRAY    = "#4a4f6a"

# ─────────────────────────────────────────────────────────────────────────────
# 0.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
data_path = ""   # set your data_path prefix here, e.g. "C:/data/"

df_universe = pd.read_parquet(data_path + "prep_data/universe_data.parquet")
MSCI_universe_membership = pd.read_parquet(
    data_path + "prep_data/MSCI_World_universe_over_time.parquet"
)
MSCI_world_data = pd.read_parquet(
    data_path + "prep_data/MSCI_World_marketCap_Vol.parquet"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# ── 1a. Parse dates ───────────────────────────────────────────────────────────
df_universe["ANNOUNCEDDATE"] = pd.to_datetime(df_universe["ANNOUNCEDDATE"])
df_universe["year"]          = df_universe["ANNOUNCEDDATE"].dt.year

MSCI_universe_membership.index = pd.to_datetime(MSCI_universe_membership.index)
MSCI_world_data["PRICINGDATE"] = pd.to_datetime(MSCI_world_data["PRICINGDATE"])
MSCI_world_data["year"]        = MSCI_world_data["PRICINGDATE"].dt.year

# ── 1b. Keep primary-flag rows only for MSCI market data ─────────────────────
msci_primary = MSCI_world_data[MSCI_world_data["TRADINGITEM_PRIMARYFLAG"] == True].copy()

# ── 1c. Build flat set of MSCI members per date ───────────────────────────────
#   MSCI_universe_membership has True/False; column names = tradingItemIds (as str)
#   We cast all column names to str for safe merging.
MSCI_universe_membership.columns = MSCI_universe_membership.columns.astype(str)

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 1
# "What fraction of M&A deals (targets + acquirers) are covered by the
#  current MSCI World universe at the time of announcement?"
# ─────────────────────────────────────────────────────────────────────────────

def msci_member_on_date(trading_item_id, announcement_date, membership_df):
    """Return True if trading_item_id was in MSCI World on or before announcement_date."""
    tid = str(trading_item_id)
    if tid not in membership_df.columns:
        return False
    # Find closest date <= announcement_date
    valid_dates = membership_df.index[membership_df.index <= announcement_date]
    if len(valid_dates) == 0:
        return False
    closest = valid_dates[-1]
    return bool(membership_df.loc[closest, tid])

# Map: for each deal, check if target & acquirer were MSCI members at announcement
df_universe["tar_tradingitemid"] = df_universe["tar_tradingitemeid"].astype(str) \
    if "tar_tradingitemeid" in df_universe.columns \
    else df_universe.get("tar_tradingitemid", pd.Series(dtype=str)).astype(str)

df_universe["acq_tradingitemid"] = df_universe["acq_tradingitemeid"].astype(str) \
    if "acq_tradingitemeid" in df_universe.columns \
    else df_universe.get("acq_tradingitemid", pd.Series(dtype=str)).astype(str)

# NOTE: Column name may vary; adapt to your actual column name.
# The screenshots show: 'acq_tradingitemid', 'tar_tradingitemid'
# We do a vectorised lookup using the membership matrix.

def batch_msci_check(trading_ids, dates, membership_df):
    """Vectorised MSCI membership check."""
    results = []
    for tid, dt in zip(trading_ids, dates):
        results.append(msci_member_on_date(tid, dt, membership_df))
    return pd.array(results, dtype=bool)

print("Checking MSCI membership for targets (this may take a moment)...")
df_universe["tar_in_msci"] = batch_msci_check(
    df_universe["tar_tradingitemid"],
    df_universe["ANNOUNCEDDATE"],
    MSCI_universe_membership
)

print("Checking MSCI membership for acquirers...")
df_universe["acq_in_msci"] = batch_msci_check(
    df_universe["acq_tradingitemid"],
    df_universe["ANNOUNCEDDATE"],
    MSCI_universe_membership
)

df_universe["either_in_msci"] = df_universe["tar_in_msci"] | df_universe["acq_in_msci"]
df_universe["both_in_msci"]   = df_universe["tar_in_msci"] & df_universe["acq_in_msci"]

# Per-year coverage rates
q1_yearly = df_universe.groupby("year").agg(
    total_deals        = ("TRANSACTIONID", "count"),
    tar_covered        = ("tar_in_msci",   "sum"),
    acq_covered        = ("acq_in_msci",   "sum"),
    either_covered     = ("either_in_msci","sum"),
    both_covered       = ("both_in_msci",  "sum"),
).reset_index()

q1_yearly["tar_pct"]    = q1_yearly["tar_covered"]    / q1_yearly["total_deals"] * 100
q1_yearly["acq_pct"]    = q1_yearly["acq_covered"]    / q1_yearly["total_deals"] * 100
q1_yearly["either_pct"] = q1_yearly["either_covered"] / q1_yearly["total_deals"] * 100
q1_yearly["both_pct"]   = q1_yearly["both_covered"]   / q1_yearly["total_deals"] * 100

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 2
# "What is the market cap distribution by year for M&A targets and acquirers?"
# ─────────────────────────────────────────────────────────────────────────────

# Market cap columns are already in df_universe: tar_marketcap, acq_marketcap
# (in USD; column name from screenshot: 'tar_marketcap', 'acq_marketcap')
# Adjust column names if needed.

tar_mcap_col = "tar_marketcap"
acq_mcap_col = "acq_marketcap"

df_tar = df_universe[["year", tar_mcap_col]].rename(columns={tar_mcap_col: "marketcap_usd"})
df_tar["role"] = "Target"
df_acq = df_universe[["year", acq_mcap_col]].rename(columns={acq_mcap_col: "marketcap_usd"})
df_acq["role"] = "Acquirer"

df_mcap_long = pd.concat([df_tar, df_acq], ignore_index=True)
df_mcap_long = df_mcap_long.dropna(subset=["marketcap_usd"])
df_mcap_long["marketcap_bn"] = df_mcap_long["marketcap_usd"] / 1e3  # USD millions → billions

years_sorted = sorted(df_mcap_long["year"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 3
# "What is total MSCI World market cap over time?"
# ─────────────────────────────────────────────────────────────────────────────

# Sum market cap across all MSCI World members by date
msci_mktcap_ts = (
    msci_primary
    .groupby("PRICINGDATE")["MARKETCAPUSD"]
    .sum()
    .rename("total_mktcap_usd")
)
msci_mktcap_ts_tn = msci_mktcap_ts / 1e12   # → trillions

# Monthly resample for smooth line
msci_mktcap_monthly = msci_mktcap_ts_tn.resample("ME").last()

# ─────────────────────────────────────────────────────────────────────────────
# QUESTION 4
# "What is the 60-day average daily trading volume for M&A targets & acquirers
#  vs MSCI World stocks, and how do they compare?"
# ─────────────────────────────────────────────────────────────────────────────

# ── 4a. MSCI World ADV (60-day rolling average at each pricing date) ──────────
msci_adv = (
    msci_primary
    .sort_values(["TRADINGITEMID", "PRICINGDATE"])
    .groupby("TRADINGITEMID")["VOLUME_USD"]
    .transform(lambda x: x.rolling(60, min_periods=30).mean())
)
msci_primary = msci_primary.copy()
msci_primary["adv60_usd"] = msci_adv

# Yearly median ADV across all MSCI stocks
msci_adv_yearly = (
    msci_primary
    .groupby("year")["adv60_usd"]
    .median()
    .rename("msci_median_adv60")
    / 1e6   # → USD millions
)

# ── 4b. M&A deal 60-day ADV at announcement ───────────────────────────────────
# For each deal, find the 60-day ADV of the target & acquirer at announcement date.
# We pull it from MSCI_world_data by matching tradingItemId and the window
# [announcement - 60 days, announcement].

def compute_deal_adv(deal_row, price_data, id_col, date_col, vol_col, window=60):
    """Compute mean daily volume for a stock over [date-window, date]."""
    tid  = str(deal_row[id_col])
    date = deal_row["ANNOUNCEDDATE"]
    start = date - pd.Timedelta(days=window)
    sub = price_data[
        (price_data["TRADINGITEMID"].astype(str) == tid) &
        (price_data["PRICINGDATE"] >= start) &
        (price_data["PRICINGDATE"] <= date)
    ]
    if len(sub) == 0:
        return np.nan
    return sub[vol_col].mean()

print("Computing 60-day ADV for M&A targets (this may take a moment)...")
df_universe["tar_adv60"] = df_universe.apply(
    lambda row: compute_deal_adv(row, msci_primary, "tar_tradingitemid", "ANNOUNCEDDATE", "VOLUME_USD"),
    axis=1
)

print("Computing 60-day ADV for M&A acquirers...")
df_universe["acq_adv60"] = df_universe.apply(
    lambda row: compute_deal_adv(row, msci_primary, "acq_tradingitemid", "ANNOUNCEDDATE", "VOLUME_USD"),
    axis=1
)

# Yearly median
tar_adv_yearly  = df_universe.groupby("year")["tar_adv60"].median()  / 1e6
acq_adv_yearly  = df_universe.groupby("year")["acq_adv60"].median()  / 1e6

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (4 questions → 4 panels in one figure)
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(22, 28))
gs  = GridSpec(4, 1, figure=fig, hspace=0.45)

# ── Q1: MSCI Coverage by year (stacked area / line) ───────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_title(
    "Q1  ·  MSCI World Coverage of M&A Deals at Announcement Date",
    fontsize=13, fontweight="bold", pad=12, color="#e8eaf0"
)

years = q1_yearly["year"].values
ax1.fill_between(years, q1_yearly["either_pct"], alpha=0.15, color=TEAL)
ax1.plot(years, q1_yearly["tar_pct"],    color=CORAL,  lw=2, marker="o", ms=4, label="Target in MSCI World")
ax1.plot(years, q1_yearly["acq_pct"],    color=AMBER,  lw=2, marker="s", ms=4, label="Acquirer in MSCI World")
ax1.plot(years, q1_yearly["either_pct"], color=TEAL,   lw=2.5, marker="^", ms=4, label="Either party in MSCI World")
ax1.plot(years, q1_yearly["both_pct"],   color=PURPLE, lw=2, marker="D", ms=4, label="Both parties in MSCI World")

# Annotate deal counts on top
for _, row in q1_yearly.iterrows():
    ax1.text(row["year"], row["either_pct"] + 1.5, str(int(row["total_deals"])),
             ha="center", fontsize=7, color="#7a7f96")

ax1.set_ylabel("% of deals with party in MSCI World", color="#c9cde0")
ax1.set_xlabel("Announcement Year")
ax1.set_ylim(0, 115)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(axis="y", alpha=0.4)
ax1.text(0.01, 0.97, "Numbers above line = total deals that year",
         transform=ax1.transAxes, fontsize=7, color="#7a7f96", va="top")

# ── Q2: Market Cap distribution – box plots by year ───────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_title(
    "Q2  ·  Market Cap Distribution by Year  —  M&A Targets vs Acquirers  (USD bn, log scale)",
    fontsize=13, fontweight="bold", pad=12, color="#e8eaf0"
)

width    = 0.35
x_pos    = np.arange(len(years_sorted))
bp_props = dict(widths=width, patch_artist=True, showfliers=False,
                medianprops=dict(color="white", lw=2))

tar_data = [df_mcap_long[(df_mcap_long["year"]==y) & (df_mcap_long["role"]=="Target")]["marketcap_bn"].dropna().values
            for y in years_sorted]
acq_data = [df_mcap_long[(df_mcap_long["year"]==y) & (df_mcap_long["role"]=="Acquirer")]["marketcap_bn"].dropna().values
            for y in years_sorted]

bp1 = ax2.boxplot(tar_data, positions=x_pos - width/2 - 0.02, **bp_props)
bp2 = ax2.boxplot(acq_data, positions=x_pos + width/2 + 0.02, **bp_props)

for patch in bp1["boxes"]: patch.set_facecolor(CORAL);  patch.set_alpha(0.6)
for patch in bp2["boxes"]: patch.set_facecolor(AMBER);  patch.set_alpha(0.6)
for w in bp1["whiskers"] + bp1["caps"]: w.set_color(CORAL);  w.set_alpha(0.7)
for w in bp2["whiskers"] + bp2["caps"]: w.set_color(AMBER);  w.set_alpha(0.7)

ax2.set_yscale("log")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(years_sorted, rotation=45, fontsize=8)
ax2.set_ylabel("Market Cap (USD bn, log scale)")
ax2.set_xlabel("Announcement Year")
ax2.legend(
    handles=[mpatches.Patch(color=CORAL, alpha=0.6, label="Target"),
             mpatches.Patch(color=AMBER, alpha=0.6, label="Acquirer")],
    loc="upper left", fontsize=9
)
ax2.grid(axis="y", alpha=0.4)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}bn"))

# ── Q3: MSCI World total market cap over time ─────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_title(
    "Q3  ·  MSCI World  —  Total Market Cap Over Time  (USD trn)",
    fontsize=13, fontweight="bold", pad=12, color="#e8eaf0"
)

ax3.fill_between(msci_mktcap_monthly.index, msci_mktcap_monthly.values,
                 alpha=0.15, color=BLUE)
ax3.plot(msci_mktcap_monthly.index, msci_mktcap_monthly.values,
         color=BLUE, lw=2.5, label="MSCI World Total Mkt Cap")

ax3.set_ylabel("Total Market Cap (USD trn)")
ax3.set_xlabel("Date")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}T"))
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.4)

# Shade recessions roughly
for start, end in [("2008-09-01", "2009-06-01"), ("2020-02-01", "2020-05-01")]:
    ax3.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#ff6b6b", alpha=0.07)
ax3.text(pd.Timestamp("2008-09-15"), msci_mktcap_monthly.max() * 0.95,
         "GFC", color=CORAL, fontsize=7)
ax3.text(pd.Timestamp("2020-02-15"), msci_mktcap_monthly.max() * 0.95,
         "COVID", color=CORAL, fontsize=7)

# ── Q4: 60-day ADV comparison ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
ax4.set_title(
    "Q4  ·  60-Day Avg Daily Trading Volume (ADV)  —  M&A Deals vs MSCI World Median  (USD mn)",
    fontsize=13, fontweight="bold", pad=12, color="#e8eaf0"
)

common_years = sorted(set(tar_adv_yearly.index) | set(acq_adv_yearly.index) | set(msci_adv_yearly.index))

ax4.plot(common_years,
         [tar_adv_yearly.get(y, np.nan)  for y in common_years],
         color=CORAL,  lw=2, marker="o", ms=4, label="M&A Target  (deal median)")
ax4.plot(common_years,
         [acq_adv_yearly.get(y, np.nan) for y in common_years],
         color=AMBER,  lw=2, marker="s", ms=4, label="M&A Acquirer  (deal median)")
ax4.plot(common_years,
         [msci_adv_yearly.get(y, np.nan) for y in common_years],
         color=BLUE,   lw=2.5, marker="^", ms=4, linestyle="--",
         label="MSCI World stock  (cross-sectional median)")

ax4.set_ylabel("60-day ADV  (USD mn, median)")
ax4.set_xlabel("Year")
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}mn"))
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.4)
ax4.set_yscale("log")

# ── Overall figure label ───────────────────────────────────────────────────────
fig.suptitle(
    "M&A Universe vs MSCI World  ·  Coverage & Liquidity Analysis",
    fontsize=17, fontweight="bold", color="#ffffff", y=0.995
)

plt.savefig("ma_universe_coverage_analysis.png", dpi=160,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n✓  Saved: ma_universe_coverage_analysis.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY: Print summary table (Q1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Q1 Summary Table ──────────────────────────────────────────────────")
print(q1_yearly[["year","total_deals","tar_pct","acq_pct","either_pct","both_pct"]]
      .to_string(index=False, float_format="{:.1f}%".format))




