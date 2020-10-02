package com.codingapi.deeplearning.demo10.learn.core;

import com.codingapi.deeplearning.demo10.learn.loss.LossFunction;
import lombok.extern.slf4j.Slf4j;

import java.io.Serializable;

/**
 * 计算代价函数的值，打印得分
 * @author lorne
 * @date 2019-10-31
 * @description 逻辑回归损失函数
 */
@Slf4j
public class NeuralListener implements Serializable {


    private LossFunction lossFunction;

    private TrainingListener[] trainingListeners;

    public NeuralListener(NeuralListener.TrainingListener... trainingListeners) {
        this.trainingListeners = trainingListeners;
    }

    public void init(LossFunction lossFunction){
        this.lossFunction = lossFunction;
    }

    public void cost(long index,double cost) {
        for (TrainingListener trainingListener:trainingListeners){
            trainingListener.done(index,cost);
        }
    }


    public  interface TrainingListener extends Serializable{
        /**
         * 执行监听业务
         * @param index   训练次数
         * @param cost   损失值
         */
        void done(long index,double cost);
    }

}
