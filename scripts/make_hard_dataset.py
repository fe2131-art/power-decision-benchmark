#!/usr/bin/env python3
"""Hard synthetic Japanese power-equipment maintenance narratives (deterministic).

This is NOT observed utility data. The labels are assigned from controlled scenario
construction, and have not been approved by a field engineer. Every example has an
explanation to support independent adjudication. No third-party dataset copied.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / 'data' / 'hard_inspection_cases.jsonl'
AUDIT = HERE / 'data' / 'hard_gold_audit.csv'
STATS = HERE / 'data' / 'hard_dataset_stats.json'

# Key = (equipment, observation). Each pairing is chosen for a plausible observation
# in a field report. The examples are fictional, not electrical operating guidance.
# (observation evidence, plausible asserted/potential cause, unobserved competing claim)
SCENARIOS = {
 'transformer': {
  'heat': ('端子部の赤外線画像で周囲より15℃高い局所温度上昇を確認した','端子締結部の接触不良','絶縁油の漏出'),
  'oil_leak': ('油入変圧器のフランジ下端に新たな油滴と床面の油染みを確認した','ガスケットの損傷','端子部の発熱'),
  'operation_failure': ('有負荷タップ切換器が切替指令に追従せず再試験でも動かなかった','タップ切換器駆動部の機械的固着','油槽の腐食'),
  'corrosion': ('変圧器の外装底部に塗膜剥離と進行した錆を確認した','外装塗膜の劣化','油量の低下'),
  'damage': ('変圧器の外装板に衝突痕と変形を目視で確認した','搬入車両との接触','通信の欠落'),
  'other': ('冷却装置付近から通常と異なる断続的な振動音を確認した','冷却ファン軸受の摩耗','油漏れ'),
 },
 'breaker': {
  'heat': ('遮断器の主回路接続端子に熱画像上の局所高温部を確認した','接続端子の締付け不足','開放失敗'),
  'operation_failure': ('VCBに開放指令を出したが接点が動かず、現場再試験でも再現した','操作ばね機構の固着','外装の腐食'),
  'corrosion': ('遮断器盤の底部金属枠に著しい赤錆を目視で確認した','盤内への浸水','遠隔信号の欠落'),
  'damage': ('遮断器の点検扉に亀裂と折損を発見した','保守搬入時の物理的接触','発熱'),
  'communication': ('遠隔監視画面で遮断器の状態信号だけが断続的に欠落した','状態信号配線の接続不良','本体の投入失敗'),
  'other': ('遮断器の操作機構から通常とは異なる金属音を繰り返し聴取した','操作機構の部品摩耗','漏油'),
 },
 'switch': {
  'heat': ('区分開閉器の接続部で熱画像に明瞭なホットスポットを確認した','接点の接触抵抗増加','通信断'),
  'oil_leak': ('油入負荷開閉器の底面から絶縁油のにじみを確認した','シール材の劣化','開閉動作の不良'),
  'operation_failure': ('負荷開閉器が投入指令を受けても動作しなかった','駆動ばねの破損','接続金具の腐食'),
  'corrosion': ('開閉器の金属筐体と取付け金具に腐食斑が広がっていた','水抜き孔閉塞による滞水','通信異常'),
  'damage': ('開閉器カバーに新しいひび割れと欠けを目視確認した','飛来物との接触','発熱'),
  'communication': ('自動開閉器の遠方操作信号が監視盤に届かない事象を記録した','通信端末の接続不良','接点破損'),
  'other': ('開閉器内部から通常聞こえない低い異音が続いていた','機構部品の摩耗','油漏れ'),
 },
 'overhead_line': {
  'heat': ('架空導体の圧縮接続部に局所発熱を赤外線点検で確認した','接続部の圧着不足','素線の切断'),
  'corrosion': ('架空線の支持金具に明瞭な腐食と赤錆を確認した','金具の防食被覆損傷','電線の発熱'),
  'damage': ('架空電線の複数素線に切れを目視で確認した','飛来物の衝突','接続部の過熱'),
  'other': ('架空電線の特定区間で通常と違う放電音を聴取した','がいし表面への汚損付着','金具の腐食'),
 },
 'meter': {
  'heat': ('スマートメーターの端子台だけが周囲より高温であると確認した','端子締付け不良','遠隔検針の欠落'),
  'corrosion': ('電力量計の金属取付け枠に進行した腐食を認めた','雨水侵入','表示の欠落'),
  'damage': ('電力量計の透明カバーに亀裂と破損を発見した','物体との接触','通信断'),
  'communication': ('電力量計の遠隔検針値が上位系に届かない事象を複数回記録した','無線通信モジュールの故障','表示異常'),
  'other': ('電力量計の液晶表示の一部が欠けていることを現場で確認した','表示基板の不具合','通信断'),
 },
 'other': {
  'heat': ('蓄電池盤の接続端子に局所的な温度上昇を確認した','接続端子の接触不良','遠隔信号断'),
  'operation_failure': ('非常用発電機の始動試験で起動しなかった','燃料供給系の閉塞','筐体の腐食'),
  'corrosion': ('受配電盤の扉ヒンジ部に著しい錆を認めた','結露による滞水','通信断'),
  'damage': ('地下電力ケーブルの外被に亀裂を確認した','掘削機材との接触','局所発熱'),
  'communication': ('遠方監視装置から中央監視へのデータ転送が繰り返し失敗した','ネットワーク端末の設定不整合','油漏れ'),
  'other': ('直流電源盤から異常な振動音を聴取した','冷却ファンの軸受摩耗','腐食'),
 }
}
ASSETS = {
 'transformer': ['配電用変圧器TR-17','変圧器 T-42','所内変圧器No.6','油入変圧器Tr-09'],
 'breaker': ['VCB-09','高圧遮断器 CB-32','遮断器No.18','真空遮断器VCB-3'],
 'switch': ['油入負荷開閉器LBS-7','区分開閉器 SW-11','自動開閉器S-28','負荷開閉器LBS-04'],
 'overhead_line': ['架空線区間L-23','架空電線 3号線','6.6kV電線L-08','架空導体 L-14'],
 'meter': ['電力量計M-61','スマートメーターNo.004','遠隔検針計器M-23','電力量計 M-109'],
 'other': ['受電設備の蓄電池盤B-2','非常用発電機 G-03','受配電盤盤内機器 P-4','地下電力ケーブル区間C-7'],
}
# Corrected below by pair for equipment='other' and for oil-filled switch etc.
OTHER_ASSET_BY_OBS = {
 'heat':'蓄電池盤B-2','operation_failure':'非常用発電機G-03',
 'corrosion':'受配電盤P-4','damage':'地下電力ケーブルC-7',
 'communication':'遠方監視装置RTU-6','other':'直流電源盤DC-3',
 'normal':'受配電盤P-4',
}
OTHER_TARGET = {
 'transformer':'遮断器CB-94','breaker':'架空電線L-18','switch':'電力量計M-87',
 'overhead_line':'開閉器SW-37','meter':'配電用変圧器TR-90','other':'高圧遮断器CB-9',
}
OTHER_NEIGHBOR_ISSUE = {
 'transformer':'開放試験の応答が遅い','breaker':'支持金具に赤錆がある',
 'switch':'遠隔検針の欠測がある','overhead_line':'投入応答が遅れる',
 'meter':'油槽下部に油染みがある','other':'開放指令への応答が遅れる',
}
CAUSE_FORMS = {
 'confirmed': [
  '対象の異常について、分解点検と再現試験により「{cause}」が原因と特定された。',
  '調査記録の最終欄に「{cause}に起因」と記載され、関連試験でも確認済み。',
  '原因分析は完了し、{cause}が本件の原因であることを確認した。',
  '速報では未確定だったが、今回の追加試験で{cause}によるものと確定した。',
 ],
 'suspected': [
  '原因候補は{cause}だが、現時点で裏付け試験はなく推定にとどまる。',
  '{cause}を疑うというコメントがある。原因の特定には至っていない。',
  '担当者は{cause}の可能性を示したが、根拠を確認できておらず暫定見解。',
  '速報の「原因確定」という表現は撤回された。{cause}が疑われる段階である。',
 ],
 'unknown': [
  '原因の記載はなく、観測事実のみを登録した。',
  '原因調査は未着手で、要因の推定も行っていない。',
  '原因は未特定。推定原因に関する記録も存在しない。',
  '別機器の故障原因は確定済みだが、この対象の原因欄は空欄である。',
 ],
}
# All observation labels are observed facts, not expert inferences from values alone.
NORMAL_FACTS = {
 'transformer':'油滴・異臭・発熱を確認できず、変圧器本体は正常と記録された',
 'breaker':'開閉試験で動作良好、発熱・損傷も認めず異常なしと記載された',
 'switch':'開閉操作と目視点検で異常なしと記録された',
 'overhead_line':'電線、支持金具、接続部に異常所見なしと確認された',
 'meter':'現地表示・通信・外観を点検し全項目異常なしと記録された',
 'other':'受配電盤の外観と動作確認で異常は認められなかった',
}


def choose_asset(e: str, o: str, k: int) -> str:
    if e == 'other':
        return OTHER_ASSET_BY_OBS[o]
    if e == 'switch' and o == 'oil_leak':
        return '油入負荷開閉器LBS-7'
    if e == 'transformer' and o == 'operation_failure':
        return '有負荷タップ切換器付き変圧器TR-29'
    return ASSETS[e][k % len(ASSETS[e])]


def narration(style: int, e: str, obs: str, cause: str, asset: str, finding: str, nonfinding: str) -> tuple[str, str]:
    neighbor = OTHER_TARGET[e]
    # All templates retain a single explicitly identified primary subject/current event.
    if style == 0:
        text = (f'定期巡視メモ／対象={asset}。隣接する{neighbor}については「{OTHER_NEIGHBOR_ISSUE[e]}との報告」が引継ぎにあるが、現地では再現せず。'
                f'今回の対象設備では、{finding}。{cause}')
        why = '対象機と別設備の未確認・否定された事象を分ける。'
        return text, why
    if style == 1:
        text = (f'週初の引継ぎ欄には{asset}で「{nonfinding}」と書かれた。'
                f'ただし本日付の確定版では旧記述は取り消し。最新の点検結果：{finding}。'
                f'{cause} 過去の速報ではなく本日付を採用すること。')
        why = '訂正された旧情報よりも現在有効な点検記録を採用する。'
        return text, why
    if style == 2:
        text = (f'記録訂正票：最初の写真ラベルは{neighbor}となっていたが、設備番号の取違えが判明。'
                f'正しくは{asset}の写真と確認された。観測事実は「{finding}」。'
                f'{cause} ただし写真ラベルの誤り自体は設備故障を意味しない。')
        why = '写真の設備番号訂正と、故障そのものの確定を混同しない。'
        return text, why
    if style == 3:
        text = (f'保全履歴の統合：{neighbor}では前月に別件対応済み。{asset}は今月の対象。'
                f'現地所見={finding}。点検者メモ「{nonfinding}は観察していない」。'
                f'原因欄の更新：{cause} 統合前の他設備の記録は本件に引き継がない。')
        why = '異なる設備・異なる時点のデータ混入を避けて根拠を読む。'
        return text, why
    raise AssertionError(style)


def normal_text(style: int, e: str, asset: str) -> tuple[str,str]:
    fact = NORMAL_FACTS[e]
    neighbor = OTHER_TARGET[e]
    if style == 0:
        return (f'本日点検対象は{asset}。前日の速報で発熱の疑いと記されたが、再検査で誤報と判明。{fact}。原因の特定対象となる異常はない。',
                '過去の疑いは棄却され、対象設備は現時点で異常なし。')
    if style == 1:
        return (f'{neighbor}では前日に「{OTHER_NEIGHBOR_ISSUE[e]}」との別件対応があったが、今回の対象は{asset}。{fact}。他設備の異常と混同しない。',
                '別設備の異常を対象に移さない。')
    if style == 2:
        return (f'前回は{asset}の動作に懸念があると通報された。本日の確認では{fact}。過去の見立ては取り下げられている。',
                '現在の正常所見を、取り下げられた速報より優先する。')
    return (f'巡視票No. {301+len(asset)}、設備：{asset}。{fact}。点検記録には同地点の旧故障例が添付されているが、対象設備の今回の故障記録ではない。',
            '今回点検した設備は異常なし、旧故障は参考添付。')

# Expert-review candidates. These are author-labelled naturalistic boundary cases.
# The cases are deliberately *not* tied to the programmatic core templates.
MANUAL = [
 ('retraction','transformer','heat','suspected','TR-4の赤外線画像に端子部の熱集中あり。速報には「原因＝接続部の緩み、確定」とあるが、最終報で締付け測定未実施と判明して確定判定を取消し。接触不良の疑いは残る。','原因確定の記載は撤回。観測は発熱、原因は疑いにとどまる。'),
 ('retraction','breaker','operation_failure','unknown','遮断器CB-14が閉極指令に無応答。一次報告の「電源故障で確定」は別案件の文面混入として削除。再点検の原因欄は未記入。','動作不良は観測。別案件の原因確定情報は使用不可。'),
 ('retraction','meter','communication','suspected','M-17の遠隔検針が3回連続欠測。通信基板不良との断定記載は撤回され、現在は電波状況起因の可能性として調査継続。','現在の根拠は通信欠落と推定原因のみ。'),
 ('retraction','switch','corrosion','unknown','開閉器SW-06外装に赤錆。塩害が原因という初稿は根拠資料がなく削除された。最終記録には錆の観測だけが残る。','削除済み推定原因を再利用しない。'),
 ('retraction','overhead_line','damage','confirmed','L-5の素線切れを確認。雷撃原因との当初見解は棄却。後日の監視映像で工事車両との接触が確認され、この接触を破断原因として確定した。','旧説は撤回、別の確定原因が最新記録にある。'),
 ('retraction','other','other','suspected','直流電源盤から異音。点検票には「冷却ファン故障確定」とあるが、後日の検証で当該記述は誤転記と判明。ファン軸受摩耗の疑いのみ記録に残る。','異音は選択肢のother、原因は未確定。'),
 ('timeline','transformer','normal','unknown','TR-33は先月油滴を検知したがパッキン交換済み。今回の巡視で油滴・熱・異音は認めず。「過去の油漏れ」は履歴であり今回の観測事象ではない。','現時点の正常を採用。'),
 ('timeline','breaker','heat','confirmed','CB-8は先月開放不良で整備済み。今朝の赤外線巡視では端子部のみ異常発熱。接触抵抗測定と増し締め前後の比較で端子の緩みが原因と確定。','今回の発熱を採用、過去の動作不良は対象外。'),
 ('timeline','switch','communication','unknown','SW-12は前日通信断の通報あり。一度回復したが、本日また遠隔状態信号の欠落を現認。原因調査は未着手。','復旧済みの過去案件と本日の通信欠落を区別する。'),
 ('timeline','meter','damage','unknown','M-8の通信障害は昨日解消。本日、外装カバーに新しい亀裂を発見した。通信は正常、カバー損傷の原因は調べていない。','現在は損傷のみ。'),
 ('timeline','overhead_line','normal','unknown','架空電線L-2の支持金具に前年度腐食があり交換済み。今回の巡視では接続部、素線、金具とも異常なしと判断された。','旧履歴ではなく現在の点検所見で分類。'),
 ('timeline','other','communication','confirmed','RTU-3は先週盤扉の腐食を補修。本日は中央監視との通信が途絶、設定差分の検証で転送設定ミスを再現し原因と確定。','対象はRTUでありequipment=other。現事象は通信障害。'),
 ('cross_asset','transformer','oil_leak','unknown','同じ柱上位置で電力量計M-4に通信断、変圧器TR-22には油受け皿への油滴を確認。点検対象欄はTR-22のみ。油滴の発生源についての検証は未実施。','他機器通信断を対象設備の異常に混入させない。'),
 ('cross_asset','breaker','corrosion','suspected','構内では開閉器SW-3が投入不良。別案件の対象CB-5は動作正常で筐体端部の錆が今回の主所見。結露影響を疑うが現場検証なし。','遮断器の腐食のみを選択し、別設備の動作不良を捨てる。'),
 ('cross_asset','switch','operation_failure','confirmed','配電所Aの記録にLBS-6の投入不能と、変圧器TR-6の油漏れが同時に載る。集計対象はLBS-6。分解で駆動ばねの破損を確認し、投入不能の原因と特定。','対象設備の指示に従い、二つの事象を混同しない。'),
 ('cross_asset','overhead_line','heat','suspected','巡視区間L-11で圧縮スリーブが局所高温。隣の開閉器は腐食しているが別票管理。圧着不足が温度上昇の要因かもしれないという所見。','熱いのは架空線接続部、隣接開閉器の腐食は別事象。'),
 ('cross_asset','meter','other','unknown','主対象M-9は液晶の上2桁が欠けて読めない。隣のVCBは開放試験不合格。計器本体の通信は成功しており、液晶異常の要因は記録なし。','液晶表示不良は分類上other、VCBは対象外。'),
 ('cross_asset','other','damage','confirmed','試掘立会票に開閉器S-6の通信断と地下ケーブルC-2の外被損傷が載る。評価対象はC-2。施工映像で掘削機のバケット接触が原因と特定済み。','ケーブルはequipment=other。'),
 ('scope','other','other','unknown','対象設備欄に「構内No.4」とだけ記載。振動か温度かは未入力で、写真も該当しない。設備分類・異常分類・原因とも判断資料がない。','設備と事象を特定できない。'),
 ('scope','transformer','other','suspected','変圧器TR-11の油槽には異常がない。冷却ファン周辺の周期的な異音だけ記録。軸受摩耗が疑われるが分解点検前。','異音はother、冷却装置でも主対象は変圧器。'),
 ('scope','breaker','communication','unknown','VCBの開閉試験は正常。一方、中央監視画面だけ状態フラグが断続的に欠落。故障箇所が盤側か通信網側かはまだ不明。','遠隔信号の欠落はcommunication。原因不明。'),
 ('scope','switch','oil_leak','confirmed','油入負荷開閉器LBS-4の床面に絶縁油のにじみ。容器からの滲出と染色確認し、シール材の劣化が原因と確定。開閉操作は良好。','油入型開閉器の漏油。'),
 ('scope','overhead_line','other','unknown','架空線L-14で異常なコロナ音が聞こえるとの現場報告。素線切断や過熱は現認されておらず、音の要因について調査結果はない。','音の事象はother。未観測の損傷を捏造しない。'),
 ('scope','meter','normal','unknown','M-2について「検針不可」の連絡が入ったが、調査で顧客システム側の集計遅延と判明。計器の現地表示・通信試験・外観はすべて正常。','異常は計器自体になく、主対象は正常。'),
 ('negation','transformer','damage','confirmed','変圧器TR-10の油漏れ・発熱は認めない。搬入時に外装パネルが変形した事実を映像で確認、輸送台車との接触が損傷原因と判断された。','否定された熱・油漏れでなく外装損傷。'),
 ('negation','breaker','normal','unknown','遮断器CB-27に「投入できない」との古い障害申告が添付。今朝の実試験は投入・遮断とも正常、熱・錆・損傷もなし。申告内容は当該機で再現しなかった。','古い申告は異常の観測ではない。'),
 ('negation','switch','heat','unknown','SW-3の通信断はネットワーク集計の誤り。現物では接点部に熱斑を確認。ただし温度上昇要因を示す試験・推定は記録されていない。','誤報の通信より実測の発熱を採用。'),
 ('negation','overhead_line','corrosion','suspected','L-21では電線の断線はなく電流値も通常範囲。支持金具の錆のみ目視確認。防食処理不足が原因かもしれないが履歴未照合。','正常な電線と腐食した支持金具を区別。'),
 ('negation','meter','heat','confirmed','M-62はデータ欠損なし・LCDも正常。ただし端子部分だけ赤外線で高温を確認。締付け状態を点検して緩みを特定、これが発熱原因と記載。','他の系統が正常でも端子過熱はある。'),
 ('negation','other','normal','unknown','受配電盤P-19には旧版に「腐食進行」の注記があるが、現場で別盤の記録と判明。対象盤の今日の外観点検は正常。','設備番号の取違えであり対象は正常。'),
 ('causal','transformer','other','unknown','TR-5から周期的な異音を確認。前日に運転負荷が変化した事実はあるが、それが異音の原因だという判断・仮説は記録されていない。','同時に起きた事柄を自動的に原因と推論しない。'),
 ('causal','breaker','operation_failure','suspected','CB-19の開放が2回遅延。同じ時間帯に制御電圧が低下した。担当は関連を疑うと明記したが、再現試験は未実施。','仮説が文章にあるのでsuspected。'),
 ('causal','switch','corrosion','unknown','開閉器SW-14に腐食を観測。降雨後という時系列情報だけがある。雨水侵入を原因とする記載はなく調査も行われていない。','雨の事実だけで雨水原因としない。'),
 ('causal','overhead_line','damage','suspected','架空線L-6に素線切れ。近くに樹木の枝があり、接触の可能性が報告されたが、痕跡調査と映像確認は未了。','枝の存在と原因確定は異なる。'),
 ('causal','meter','communication','confirmed','M-5の通信欠損。通信モジュールの交換試験で異常が解消し、元のモジュールで再発することを確認。原因を同モジュール不良と確定した。','再現性を根拠に原因確定と明記。'),
 ('causal','other','heat','unknown','蓄電池盤B-7の端子で温度上昇を検知。運転電流が大きい時間帯だったが因果関係の記録はない。原因欄は空欄。','同時相関から故障原因を断定しない。'),
 ('mixed','transformer','oil_leak','suspected','TR-21の接地値正常、発熱所見なし、油面計の表示も正常。しかしフランジ外周に新しい油滴を確認。ガスケット劣化が疑われるものの未分解。','正常項目が複数あっても観測した油滴を採用。'),
 ('mixed','breaker','other','confirmed','遮断器CB-17は投入遮断の成功率に異常なし。操作機構から引っ掛かる金属音が出ており、分解で軸部摩耗を確認し異音原因と特定。','音はoperation_failureではなくother。'),
 ('mixed','switch','damage','unknown','SW-30の遠隔制御と温度は正常。外装樹脂部の欠けを点検写真で確認したが、損傷のきっかけは不明。','現在実在する物理損傷に着目する。'),
 ('mixed','overhead_line','normal','unknown','L-3の過去の温度アラームはセンサーの検出誤りと確認。今回の現地確認では電線、接続金具とも異常所見なし。','アラームを事実と誤認しない。'),
 ('mixed','meter','corrosion','confirmed','M-7の通信・表示は正常だが、取付金具に錆を確認。漏水痕と点検記録の突合で取付部への浸水が原因と確定された。','通信・表示正常は錆の否定にならない。'),
 ('mixed','other','operation_failure','suspected','非常用発電機G-8が起動試験で無応答。盤内通信と表示は正常。燃料供給不足が疑われるが圧力測定未実施。','別系統正常でも起動不能は観測。'),
]


def build_records() -> list[dict]:
    records = []
    n=0
    for e, obsmap in SCENARIOS.items():
        for o, (finding, probable_cause, nonfinding) in obsmap.items():
            # All evidence conditions are explicitly stated; labels are not computed from a keyword match.
            for c in ('confirmed', 'suspected', 'unknown'):
                for style in range(4):
                    asset = choose_asset(e,o,style)
                    cause = CAUSE_FORMS[c][style].format(cause=probable_cause)
                    text, reason = narration(style,e,o,cause,asset,finding,nonfinding)
                    n+=1
                    records.append({
                        'id':f'H-{n:04}', 'group':('cross_asset','timeline','correction','dense')[style],
                        'text': text, 'gold': {'equipment':e, 'observation':o,'cause_status':c},
                        'family_id': f'{e}.{o}', 'variant':f'template_{style}', 'difficulty':'hard',
                        'rationale': reason + (' 原因が確定。' if c=='confirmed' else ' 原因が推定。' if c=='suspected' else ' 原因の根拠なし。'),
                        'gold_origin':'controlled_synthetic',
                    })
    for e in SCENARIOS:
        for style in range(4):
            asset = choose_asset(e,'normal',style)
            text,why = normal_text(style,e,asset)
            n+=1
            records.append({'id':f'H-{n:04}', 'group':'resolved' if style in (0,2) else 'cross_asset',
                            'text':text, 'gold':{'equipment':e,'observation':'normal','cause_status':'unknown'},
                            'family_id':f'{e}.normal','variant':f'normal_{style}', 'difficulty':'hard',
                            'rationale':why,'gold_origin':'controlled_synthetic'})
    for group,e,o,c,text,why in MANUAL:
        n+=1
        records.append({'id':f'H-{n:04}', 'group':group,'text':text,
                        'gold':{'equipment':e,'observation':o,'cause_status':c},
                        'family_id':f'authored.{group}', 'variant':'hand_authored',
                        'difficulty':'very_hard', 'rationale':why,'gold_origin':'author_labelled_synthetic'})
    # Round-robin interleave group strata; --limit must not always select a single family.
    groups={g: [r for r in records if r['group']==g] for g in sorted({r['group'] for r in records})}
    shuffled=[]
    while any(groups.values()):
        for g in groups:
            if groups[g]: shuffled.append(groups[g].pop(0))
    return shuffled


def render() -> tuple[str, str, str]:
    rows=build_records()
    dataset=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows)
    import io
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=['id','group','difficulty','family_id','variant','equipment','observation','cause_status','rationale','gold_origin'], lineterminator='\n')
    writer.writeheader()
    for r in rows:
        writer.writerow({k:(r['gold'][k] if k in r['gold'] else r.get(k,'')) for k in writer.fieldnames})
    stats={
        'count':len(rows), 'synthetic':True, 'human_expert_reviewed':False,
        'gold_policy':'target current observation; ignore other devices, retracted notices, historical resolved events; cause evidence only',
        'difficulty':dict(sorted(Counter(r['difficulty'] for r in rows).items())),
        'groups':dict(sorted(Counter(r['group'] for r in rows).items())),
        'labels':{f:dict(sorted(Counter(r['gold'][f] for r in rows).items())) for f in ('equipment','observation','cause_status')},
        'sha256_dataset':hashlib.sha256(dataset.encode('utf-8')).hexdigest(),
    }
    return dataset,buffer.getvalue(),json.dumps(stats,ensure_ascii=False,indent=2)+'\n'


def main():
    OUT.parent.mkdir(exist_ok=True,parents=True)
    a,b,c=render()
    OUT.write_text(a,'utf-8')
    AUDIT.write_text(b,'utf-8-sig')
    STATS.write_text(c,'utf-8')
    print(f'created {len(a.splitlines())} hard cases: {OUT}')
    print(f'sha256 {hashlib.sha256(a.encode()).hexdigest()}')

if __name__=='__main__':
    main()
