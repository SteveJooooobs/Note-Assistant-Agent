如果没有KV Cache，生成新Token时，会把整个序列重新送进Transformer

于是导致了：
Q1~Qn
K1~Kn
V1~Vn
全部重新计算一遍

但实际上：
历史位置对应的输出已经确定，不会因为未来Token出现而改变
因此，历史位置的Q计算完全是在重复劳动

KV Cache的做法是：

保留历史Token已经算好的K和V，下一轮只计算新位置的Q、K、V。

Q_new  只与  历史K/V + 当前K/V做一次Attention
这样就避免了重新计算历史Q和历史Attention

所以更本质的说法是：
KV缓存把Transformer从"全序列重算"变成"增量计算"