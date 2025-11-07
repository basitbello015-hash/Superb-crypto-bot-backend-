// static/dashboard.js — simple helper to fetch overview from API
(async function () {
  async function fetchOverview() {
    try {
      // Try API endpoint commonly used in earlier conversation
      const res = await fetch('/api/dashboard/overview');
      if (!res.ok) throw new Error('no overview API');
      return await res.json();
    } catch (err) {
      // Fallback: try retrieving from /api/dashboard (if you have a different endpoint)
      try {
        const res = await fetch('/api/dashboard');
        if (!res.ok) throw new Error('no fallback');
        return await res.json();
      } catch (err2) {
        console.warn('Overview fetch failed:', err, err2);
        return null;
      }
    }
  }

  function setCards(data) {
    if (!data) return;
    const profit = data.profit ?? data.total_profit ?? 0;
    const openTrades = data.openTrades ?? data.active_trades ?? 0;
    const balance = data.balance ?? data.total_balance ?? 0;
    const daily = data.dailyChange ?? data.win_rate ?? 0;

    const elProfit = document.getElementById('card-profit');
    const elOpen = document.getElementById('card-open');
    const elBalance = document.getElementById('card-balance');
    const elDaily = document.getElementById('card-daily');

    if (elProfit) elProfit.textContent = profit;
    if (elOpen) elOpen.textContent = openTrades;
    if (elBalance) elBalance.textContent = '$' + (Number(balance).toFixed ? Number(balance).toFixed(2) : balance);
    if (elDaily) elDaily.textContent = (daily) + (String(daily).includes('%') ? '' : '%');
  }

  async function tick() {
    const d = await fetchOverview();
    if (d) setCards(d);
  }

  // initial tick and periodic refresh
  document.addEventListener('DOMContentLoaded', tick);
  setInterval(tick, 5000);
})();
