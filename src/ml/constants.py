# NOTE: Loose variables, and for now, disorganized (We need this to clean some scripts)
# NOTE: # I use gemini code assist to generate an HTML -- it's not my core...

BADGE_HTML = ' <span class="badge">✦ best</span>'

ROW_TEMPLATE = """
            <tr {style}>
                <td>{epoch}{best_badge}</td>
                <td>{train_loss:.4f}</td>
                <td>{val_loss:.4f}</td>
                <td>{auc_str}</td>
                <td>{epoch_time}</td>
                <td>{gpu_mb}</td>
            </tr>"""

CLASS_ROW_TEMPLATE = """
                <div class="class-row">
                    <span class="class-name">{label}</span>
                    <div class="bar-wrap">
                        <div class="bar" style="width:{pct}%;background:{color}"></div>
                    </div>
                    <span class="auc-val" style="color:{color}">{auc:.4f}</span>
                </div>"""

WORST_CLASSES_SECTION_TEMPLATE = """
<div class="section">
    <div class="section-title">Top-10 Worst AUC Classes (last epoch)</div>
    {content}
</div>"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DeepWetlands — Training Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #0a0c14;
    --surface: #12151f;
    --surface2: #1a1e2e;
    --border: #252a3d;
    --train: #4f9cf9;
    --val: #f97316;
    --accent: #a78bfa;
    --good: #34d399;
    --bad: #f87171;
    --text: #e2e8f0;
    --sub: #64748b;
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    padding: 0;
  }}

  .hero {{
    background: linear-gradient(135deg, #0f1628 0%, #1a0f2e 50%, #0f1628 100%);
    border-bottom: 1px solid var(--border);
    padding: 48px 64px 40px;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(167,139,250,0.12) 0%, transparent 70%);
    pointer-events: none;
  }}
  .hero-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .hero h1 {{
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 8px;
  }}
  .hero h1 span {{ color: var(--accent); }}
  .hero-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--sub);
    margin-top: 16px;
  }}

  .container {{ max-width: 1200px; margin: 0 auto; padding: 48px 64px; }}

  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 48px;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
  }}
  .kpi::after {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }}
  .kpi.train::after  {{ background: var(--train); }}
  .kpi.val::after    {{ background: var(--val); }}
  .kpi.auc::after    {{ background: var(--good); }}
  .kpi.time::after   {{ background: var(--accent); }}
  .kpi-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--sub);
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  .kpi-value {{
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
  }}
  .kpi.train .kpi-value  {{ color: var(--train); }}
  .kpi.val .kpi-value    {{ color: var(--val); }}
  .kpi.auc .kpi-value    {{ color: var(--good); }}
  .kpi.time .kpi-value   {{ color: var(--accent); }}
  .kpi-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--sub);
    margin-top: 8px;
  }}

  .section {{ margin-bottom: 48px; }}
  .section-title {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--sub);
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}

  table {{ width: 100%; border-collapse: collapse; }}
  thead tr {{
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }}
  th {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--sub);
    padding: 12px 16px;
    text-align: left;
  }}
  td {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  tbody tr:hover {{ background: var(--surface2); }}
  .badge {{
    font-size: 9px;
    background: rgba(52,211,153,0.15);
    color: var(--good);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 4px;
    padding: 2px 6px;
    margin-left: 6px;
    vertical-align: middle;
    letter-spacing: 1px;
  }}

  .plots-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}
  .plot-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }}
  .plot-card img {{ width: 100%; display: block; }}
  .plot-card.wide {{ grid-column: 1 / -1; }}

  .class-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
  }}
  .class-name {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    min-width: 140px;
    color: var(--sub);
  }}
  .bar-wrap {{
    flex: 1;
    background: var(--surface2);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
  }}
  .bar {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }}
  .auc-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    min-width: 52px;
    text-align: right;
  }}

  footer {{
    text-align: center;
    padding: 32px;
    color: var(--sub);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-label">BirdCLEF+ Pantanal 2026</div>
  <h1>DeepWetlands<span>Pulse</span><br>Training Report</h1>
  <div class="hero-meta">
    Generated at {timestamp} &nbsp;·&nbsp;
    {num_epochs} epochs &nbsp;·&nbsp;
    {num_classes} classes &nbsp;·&nbsp;
    Total Duration: {total_time}
  </div>
</div>

<div class="container">

  <div class="kpi-grid">
    <div class="kpi train">
      <div class="kpi-label">Best Train Loss</div>
      <div class="kpi-value">{best_train_loss}</div>
      <div class="kpi-sub">epoch {best_train_epoch}</div>
    </div>
    <div class="kpi val">
      <div class="kpi-label">Best Val Loss</div>
      <div class="kpi-value">{best_val_loss}</div>
      <div class="kpi-sub">epoch {best_val_epoch}</div>
    </div>
    <div class="kpi auc">
      <div class="kpi-label">Best Macro-AUC</div>
      <div class="kpi-value">{best_macro_auc}</div>
      <div class="kpi-sub">val set</div>
    </div>
    <div class="kpi time">
      <div class="kpi-label">Total Time</div>
      <div class="kpi-value">{total_time}</div>
      <div class="kpi-sub">~{avg_time_epoch} / epoch</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">History per Epoch</div>
    <table>
      <thead>
        <tr>
          <th>Epoch</th>
          <th>Train Loss</th>
          <th>Val Loss</th>
          <th>Macro-AUC</th>
          <th>Time</th>
          <th>GPU</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Plots</div>
    <div class="plots-grid">
      <div class="plot-card wide"><img src="loss_curve.png" alt="Loss Curve"></div>
      <div class="plot-card"><img src="top_k_errors.png" alt="Top-K Errors"></div>
      <div class="plot-card"><img src="gpu_memory.png" alt="GPU Memory"></div>
    </div>
  </div>

  {worst_classes_section}

</div>

<footer>DeepWetlands · BirdCLEF+ 2026 · automatically generated by training_logger.py</footer>
</body>
</html>"""