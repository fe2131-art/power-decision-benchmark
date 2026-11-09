#!/usr/bin/env python3
"""人工合成の電力設備点検記録。実データではない。"""
from pathlib import Path
import json
import sys

# (group, equipment, observation, cause_status, text)
CASES = [
('clear','transformer','heat','suspected','配電用変圧器の外箱で局所的な温度上昇を確認。端子部の接触抵抗増大が疑われるが、原因は未確定。'),
('clear','transformer','oil_leak','confirmed','柱上変圧器の下部に絶縁油のにじみを確認。点検によりパッキンの損傷が原因と確定。'),
('clear','breaker','operation_failure','confirmed','VCBが開放指令後も動作しなかった。調査で操作回路ヒューズの断線が原因と判明した。'),
('clear','switch','corrosion','unknown','区分開閉器の金属ケースに広範な錆が認められた。原因調査は未実施。'),
('clear','overhead_line','damage','suspected','架空電線の一部素線の切断を発見。鳥害の可能性があるものの、調査中。'),
('clear','meter','communication','unknown','スマートメーターから遠隔検針データを取得できず、通信エラーを記録。原因は記載されていない。'),
('clear','transformer','normal','unknown','配電用変圧器を定期巡視。油漏れ、異音、発熱とも認められず、異常なし。'),
('clear','breaker','heat','suspected','遮断器の接続端子で局所発熱を検出。増し締め不足の疑いがある。'),
('clear','switch','operation_failure','confirmed','LBSの投入動作に失敗。調査の結果、ばね機構の破損が原因と確認された。'),
('clear','overhead_line','corrosion','unknown','架空線の支持金具に著しい腐食を観察。現場報告には原因の記載なし。'),
('clear','meter','damage','confirmed','電力量計のカバーに亀裂。落下物との接触が監視画像で確認され、破損原因と特定。'),
('clear','transformer','oil_leak','suspected','油入変圧器で油滴を確認。シール劣化が原因の可能性があるため継続調査する。'),
('clear','breaker','normal','unknown','遮断器の開閉試験は規定どおりで、異常なし。'),
('clear','switch','communication','unknown','配電用自動開閉器の遠隔監視信号が断続的に欠落。原因は調査中。'),
('clear','overhead_line','heat','confirmed','架空電線接続部で温度異常を記録。分解調査により接続金具の緩みが原因と確定。'),
('clear','meter','normal','unknown','電力量計の通信および表示を点検。いずれも正常で異常を認めず。'),
('negation','transformer','oil_leak','unknown','配電用変圧器の発熱は確認されなかったが、タンク底部に油漏れを確認した。原因未特定。'),
('negation','breaker','corrosion','unknown','遮断器は動作良好。操作不良は認めないが、外部鉄部には錆が発生している。'),
('negation','switch','normal','unknown','開閉器の前回指摘した腐食箇所は補修済み。今回の巡視では新たな異常を認めず。'),
('negation','overhead_line','damage','unknown','架空線の接続部温度は正常。ただし別箇所の電線表面に亀裂が見つかった。原因不明。'),
('negation','meter','communication','suspected','電力量計の機器表示は正常だが、上位システムへの通信が失敗。周辺電波環境の影響が疑われる。'),
('negation','transformer','normal','unknown','油入変圧器の油漏れ疑いは再点検で否定された。発熱もなく現時点では異常なし。'),
('negation','breaker','operation_failure','suspected','遮断器に発熱はないが、開放動作に遅れが観測された。電圧低下の影響かもしれない。'),
('negation','switch','oil_leak','unknown','絶縁油封入型の開閉器に油のにじみ。動作試験では異常はなかった。原因の記載なし。'),
('negation','overhead_line','normal','unknown','架空電線は雷雨後に点検したが、断線・発熱・腐食などの異常は認めない。'),
('negation','meter','damage','unknown','スマートメーターの通信異常は復旧済み。今回確認した異常は筐体のひび割れのみ。'),
('negation','transformer','corrosion','suspected','変圧器の外装に錆を確認。油漏れはない。海塩粒子が腐食要因の可能性がある。'),
('negation','breaker','normal','unknown','遮断器の動作不良という事前通報があったが、現地再現試験では正常動作し異常なし。'),
('negation','switch','heat','confirmed','開閉器で通信欠落の報告は誤検知だった。一方、接点に発熱があり、接触不良が原因と確認した。'),
('negation','overhead_line','corrosion','confirmed','架空線の接続金具に錆。断線はなかった。調査で防食被覆の剥離が原因と確認。'),
('hypothesis','transformer','other','suspected','変圧器から異音が出ている。鉄心の緩みの可能性があるが、振動計測は未実施。'),
('hypothesis','breaker','other','unknown','遮断器から普段と異なるうなり音がある。油漏れや発熱はなく、原因の見通しはない。'),
('hypothesis','switch','operation_failure','unknown','開閉器の投入操作ができない。故障要因の候補は挙げられておらず、切り分け中。'),
('hypothesis','overhead_line','other','suspected','架空電線が風で大きく振れているとの通報。支持金具の固定不足が原因かもしれないが、損傷自体は未確認。'),
('hypothesis','meter','communication','confirmed','遠隔検針に失敗。電力量計の通信モジュール故障を交換前後の試験で再現し、原因と判定。'),
('hypothesis','transformer','heat','unknown','変圧器の放熱器表面が通常より高温と記録された。原因候補は未記載。'),
('hypothesis','breaker','operation_failure','suspected','VCBのトリップ試験で応答が遅い。操作電源低下の可能性を疑うが確証なし。'),
('hypothesis','switch','corrosion','confirmed','開閉器の筐体で著しい錆を確認。調査で水抜き孔閉塞による滞水が原因と確定。'),
('hypothesis','overhead_line','damage','confirmed','架空導体に断線を確認。点検の結果、外力接触を示す付着物と監視映像から原因を特定。'),
('hypothesis','meter','heat','unknown','電力量計の端子台に局所発熱を認めた。温度のみ観測しており原因は未調査。'),
('out_of_scope','other','other','suspected','受変電所の蓄電池で電圧低下が発生。セル内部抵抗の増加が疑われている。'),
('out_of_scope','other','other','confirmed','非常用発電機の始動不良。燃料系配管の閉塞が原因と整備で確定した。'),
('out_of_scope','other','other','unknown','巡視報告に「設備Aで様子が変」とだけあり、対象設備も異常内容も特定できない。'),
('out_of_scope','other','normal','unknown','ケーブル洞道の照明と換気設備を点検し、異常なし。'),
('out_of_scope','other','corrosion','unknown','配電盤の筐体に腐食を確認。原因の記述はない。'),
('out_of_scope','other','communication','suspected','遠方監視装置の通信断を確認。ネットワーク機器の設定変更が原因かもしれない。'),
('out_of_scope','other','damage','unknown','地下電力ケーブルの外被に亀裂が見られたが、劣化原因は特定されていない。'),
('out_of_scope','other','normal','unknown','外灯の設備点検が完了。異常所見なし。'),
]

def main():
    default=Path(__file__).resolve().parents[1]/'data'/'legacy'/'synthetic_inspection_cases.jsonl'
    out=Path(sys.argv[1]) if len(sys.argv)>1 else default
    # 先頭から数件だけでも4つの難易度を含むよう、グループ別にラウンドロビンで並べる。
    from collections import defaultdict
    grouped=defaultdict(list)
    for rec in CASES:
        grouped[rec[0]].append(rec)
    ordered=[]
    while any(grouped.values()):
        for group in ('clear','negation','hypothesis','out_of_scope'):
            if grouped[group]:
                ordered.append(grouped[group].pop(0))
    with out.open('w',encoding='utf-8') as f:
        for i,(group,e,o,c,text) in enumerate(ordered,1):
            item={'id':f'PS-{i:03}', 'group':group, 'text':text, 'gold':{'equipment':e,'observation':o,'cause_status':c}}
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print('wrote',len(CASES),'records to',out)

if __name__=='__main__':
    main()
