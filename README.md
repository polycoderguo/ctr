# ctr
experiments of ctr prediction algorithm 

## 特征工程

数据的特征有以下trick：

* `site_id == '85f751fd'`是APP数据
* `device_id == 'a99f214a'`是匿名数据，用`device_ip + device_model`代替
* app的数据和site的数据需要分开使用, 都有id、domain、category三个属性

最后采用的feature:

* 媒体特征：`pub_id、pub_domain、pub_category`
* 广告位特征：`banner_pos`
* 用户特征：`device_model`、`device_conn_type`
* 盲特征：`C14、C17、C20、C21`
* 时间特征：`hour`
* 统计特征：
    * `device_ip_count 、device_id_count`: > 1000时为值， 否则为出现的次数
    * `smooth_user_hour_count`: 当前小时`user_id`出现的次数， >30 次则统一一个数字
    * 如果用户出现总次数>30，取用户出现总次数`user_count`，不然取用户出现出`user-count + user-click-history`，`user-click-history`去用户前一个小时最近4次的点击记录，例如`0100`（对比赛预测无用，会直接退化成`user_count`？）