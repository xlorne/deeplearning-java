package com.codingapi.deeplearning.demo07.learn;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 计算代价函数的值，打印得分
 * @author lorne
 * @date 2019-10-31
 * @description 逻辑回归损失函数
 */
@Slf4j
public class NeuralListener {


    private LossFunction lossFunction;

    private TrainingListener[] trainingListeners;

    public NeuralListener(NeuralListener.TrainingListener... trainingListeners) {
        this.trainingListeners = trainingListeners;
    }

    public void init(LossFunction lossFunction){
        this.lossFunction = lossFunction;
    }

    public void cost(long index,INDArray predict, INDArray y) {
        double sum =  lossFunction.score(predict,y);
        for (TrainingListener trainingListener:trainingListeners){
            trainingListener.done(index,sum);
        }
    }


    public  interface TrainingListener{
        /**
         * 执行监听业务
         * @param index   训练次数
         * @param sum   损失值
         */
        void done(long index,double cost);
    }

}
