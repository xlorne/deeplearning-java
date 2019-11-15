package com.codingapi.deeplearning.demo06.learn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author lorne
 * @date 2019-11-15
 * @description J(\theta )=-\frac{1}{m} \sum_{i=1}^m y^{(i)}log(h_{(\theta )}(x^{i}))+(1-y^{i})log(1-h_{\theta }(x^{i}))
 */
public class LogisticRegressionLossFunction implements LossFunction {


    @Override
    public INDArray score(INDArray predict, INDArray y) {
        INDArray first = y.mul(-1).mul(Transforms.log(predict));
        INDArray second = y.rsub(1).mul(Transforms.log(predict.rsub(1)));
        INDArray cost = first.sub(second);
        INDArray sum = Nd4j.sum(cost.div(y.rows()));
        return sum;
    }
}
