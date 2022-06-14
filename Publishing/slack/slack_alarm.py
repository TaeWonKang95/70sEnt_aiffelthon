import os, sys
from slacker import Slacker
import requests
import datetime
import pandas as pd


sys.path.append(os.path.abspath('/content/drive/MyDrive/70sEnt/'))
from Model_Build import TFT_Model_Build

def make_slack_message(ts_df, predict_hour, new_raw_predictions, region_list, percentage):
    region_ts_df_list = [east_ts_df, north_ts_df, mid_ts_df, south_ts_df, ulju_ts_df]
    predict = new_raw_predictions['prediction'][:, :, 3].numpy()
    forecast_date = pd.to_datetime(
        ts_df.iloc[ts_df['time_idx'].max() - 24 * 7 + predict_hour + 1]['datetime']).strftime('%Y-%m-%d')

    print_string = '예측 날짜 : {}'.format(forecast_date)

    for g_idx, g_name in enumerate(region_list):
        region_ts_df = region_ts_df_list[g_idx]
        print_string += '\n\n< ' + g_name + ' >\n'

        if predict[g_idx].sum() < region_ts_df[(region_ts_df['datetime'].dt.month == int(forecast_date[5:7]))
                                               & (region_ts_df['weekday'] == int(
            region_ts_df[region_ts_df['datetime'] == forecast_date]['weekday']))][
            'socar_count'].mean() * 24 * percentage:
            print_string += '전 연도 동월 동요일 평균 수요 보다 ' + str(percentage * 100) + '% 이하로 감소 예상 -- 쿠폰발행 필요\n\n'

            for i in range(24):
                if predict[g_idx][i] < region_ts_df[(region_ts_df['datetime'].dt.month == int(forecast_date[5:7])) &
                                                    (region_ts_df['weekday'] == datetime.date.weekday(
                                                        pd.to_datetime(forecast_date))) &
                                                    (region_ts_df['hour'] == str(i))]['socar_count'].mean() * (
                        percentage - 0.1):
                    print_string += '- ' + str(i) + '시 - \n예상 {}대 운행, 전년도 동월 평균 대비 {:.2f} % 예상'.format(
                        round(predict[g_idx][i]),
                        predict[g_idx][i] /
                        region_ts_df[(region_ts_df['datetime'].dt.month == int(forecast_date[5:7])) &
                                     (region_ts_df['weekday'] == datetime.date.weekday(pd.to_datetime(forecast_date))) &
                                     (region_ts_df['hour'] == str(i))]['socar_count'].mean() * 100)
                    print_string += '\n'

                    print_string += '(예상 가동율 : {:.2f} %, 전년도 동월 평균 가동율 : {:.2f} %)'.format(
                        predict[g_idx][i] / region_ts_df['socar_count'].max() * 100,
                        region_ts_df[(region_ts_df['datetime'].dt.month == int(forecast_date[5:7])) &
                                     (region_ts_df['weekday'] == datetime.date.weekday(pd.to_datetime(forecast_date))) &
                                     (region_ts_df['hour'] == str(i))]['socar_count'].mean() / region_ts_df[
                            'socar_count'].max() * 100)
                    print_string += '\n\n'

        print_string += '----------------------------------------------------'
    return print_string


 def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )
    print(response)


