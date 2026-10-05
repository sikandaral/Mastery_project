"""Make proposal figures from the computed JSON, with captions in figures/captions.md."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

core=json.loads((ROOT/'reports/eda_metrics.json').read_text())
external=json.loads((ROOT/'reports/external_metrics.json').read_text())
figdir=ROOT/'figures'
figdir.mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize':(7.2,3.9),'font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.dpi':180})
captions=[]


def save(name,caption):
    plt.tight_layout()
    plt.savefig(figdir/f'{name}.png',bbox_inches='tight')
    plt.close()
    captions.append((name,caption))


def yearmonth(value):
    return datetime.fromisoformat(value).replace(tzinfo=None)


milestones=[(datetime(1997,9,1),'launch'),(datetime(1999,11,1),'media'),
            (datetime(2003,2,18),'v3'),(datetime(2008,9,1),'movie adding'),
            (datetime(2014,11,1),'v4')]


fig,axes=plt.subplots(1,2,figsize=(9,3.6))
for ax,key,label in [(axes[0],'user_histogram','Users'),(axes[1],'movie_histogram','Rated movies')]:
    d=core[key]
    ax.bar([int(v['log10_bin']) for v in d],[v['users' if key=='user_histogram' else 'movies'] for v in d],color='#235789')
    ax.set_yscale('log');ax.set_xlabel('log10 ratings count, rounded down');ax.set_ylabel(f'Number of {label.lower()}')
    ax.set_title(label)
save('01_activity_long_tail','Rating activity is highly uneven: most rated movies have few ratings, while some users rate thousands.')

monthly=core['monthly'];dates=[yearmonth(v['month_date']) for v in monthly]
plt.figure()
plt.plot(dates,[v['ratings'] for v in monthly],lw=1,color='#235789')
for date,label in milestones:
    plt.axvline(date,color='#c05a2d',alpha=.55,lw=.8)
    plt.text(date,plt.ylim()[1]*.95,label,rotation=90,va='top',fontsize=7)
plt.ylabel('Ratings entered per month');plt.xlabel('Calendar month (UTC)')
plt.title('Rating volume across platform history')
save('02_monthly_ratings','Monthly rating volume changes across the platform history, including substantial activity before the documented launch.')

plt.figure()
plt.plot(dates,[100*v['half_share'] for v in monthly],lw=1,color='#7d4e9d')
for date,label in [(datetime(2003,2,18),'v3 scale'),(datetime(2014,11,1),'v4')]:
    plt.axvline(date,color='#c05a2d',alpha=.7,lw=.9)
    plt.text(date,plt.ylim()[1]*.92,label,rotation=90,va='top',fontsize=8)
plt.ylabel('Ratings with half star value (%)');plt.xlabel('Calendar month (UTC)')
plt.title('Half star use does not change uniformly after v3')
save('03_half_star_timeline','No half stars appear before the documented v3 change, but the share varies sharply after 2003.')

monthly_cohorts=[v for v in core['first20_monthly'] if '2012-'<=v['cohort_month'][:5]<='2016-']
plt.figure()
plt.plot([yearmonth(v['cohort_month']) for v in monthly_cohorts],[100*v['prior_top1_share'] for v in monthly_cohorts],marker='o',markersize=2.5,color='#235789')
plt.axvline(datetime(2014,11,1),color='#c05a2d',alpha=.8,lw=1,label='v4 documented month')
plt.ylabel('First 20 in prior year top 1% (%)');plt.xlabel('Month of user first rating (UTC)')
plt.title('A sharp cohort shift in November 2014')
plt.legend(frameon=False)
save('04_first20_prior','The prior-year top-1% share rises from 15.9% for October 2014 starters to 57.1% for November starters, coinciding with the documented v4 month.')

eras=['pre-v3','v3','v4+'];prior={v['era']:v for v in core['first20_prior_stable']};glob={v['era']:v for v in core['first20_global_stable']}
fig,ax=plt.subplots()
x=range(3);ax.bar([i-.18 for i in x],[100*prior[e]['prior_top1_share'] for e in eras],.36,label='Prior-year top 1%',color='#235789')
ax.bar([i+.18 for i in x],[100*glob[e]['top1_share'] for e in eras],.36,label='Full-data top 1%',color='#d59a49')
ax.set_xticks(list(x),eras);ax.set_ylabel('Share of first 20 ratings (%)');ax.set_title('Popularity definition changes the answer')
ax.legend(frameon=False)
save('05_popularity_sensitivity','Using full-data popularity substantially raises measured concentration because it includes future ratings.')

catalog=core['catalog']
plt.figure()
plt.bar([v['entry_year'] for v in catalog],[v['entered'] for v in catalog],color='#3b8872')
plt.axvline(2008.75,color='#c05a2d',label='Member movie adding (Sep 2008)')
plt.axvline(2014.83,color='#765aa8',label='v4 (Nov 2014)')
plt.xlabel('Year of first MovieLens rating (UTC)');plt.ylabel('Movies first rated')
plt.title('First rating as a catalog entry proxy');plt.legend(frameon=False,fontsize=8)
save('06_catalog_entry','The number of newly rated titles rises after catalog policy changes, but first rating is only a proxy for catalog entry.')

q=external['quadrants'];groups=['ML popular / IMDb popular','ML popular / IMDb tail','ML tail / IMDb popular','ML tail / IMDb tail']
counts=[]
for label in groups:
    ml,imdb=[s.strip() for s in label.split('/')]
    counts.append(next(v['movies'] for v in q if v['ml_group']==ml and v['imdb_group']==imdb))
plt.figure()
bars=plt.barh(groups[::-1],counts[::-1],color=['#a7b8ca','#d59a49','#d59a49','#235789'])
plt.xscale('log');plt.xlabel('Matched movies (log scale)');plt.title('MovieLens and current IMDb popularity')
for b,n in zip(bars,counts[::-1]):plt.text(b.get_width()*1.03,b.get_y()+b.get_height()/2,f'{n:,}',va='center',fontsize=8)
save('07_external_quadrants','Most matched titles fall below both top-decile cutoffs, while roughly two thousand fall in each mixed-popularity cell.')

sessions=[core['sessions_30'],core['sessions_60']]
plt.figure(figsize=(6.5,3.5))
plt.bar(['30 min','60 min'],[100*s['ratings_in_50plus']/core['sizes']['ratings'] for s in sessions],color=['#235789','#6d9fbd'])
plt.ylim(0,100);plt.ylabel('Ratings in sessions of 50+ (%)');plt.xlabel('Session gap threshold')
plt.title('Large batch-entry sessions are common')
save('08_session_sensitivity','About two thirds of ratings occur in sessions of at least 50 entries under either session gap rule.')

(figdir/'captions.md').write_text('# Figure captions\n\n'+''.join(f'- **{name}.png** — {caption}\n' for name,caption in captions))
