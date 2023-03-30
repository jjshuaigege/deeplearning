import numpy as np
import copy
import functions as F
class MyNeuralNetwork:

    #所用的一些hyperparameter介绍：各层神经单元数，initialize为初始化方法，可取"Xavier"和"kaiming",激活函数有(sigmoid,tanh,relu)，学习率，batchsize。
    # update为更新方式——有MBGD和Adam两种，默认="MBGD"
    #lambd用于L2正则化
    #keep_prob是个向量，代表各层（包括输入输出）的dropout概率
    #beta_1和beta_2,epsilon用于adam算法。
    # batch_normalize=True时，每个隐藏层后面会接一层bn层,beta_bn用于指数加权平均计算均值和方差，以将二者用于测试阶段。
    def __init__(self,layer_vec,*,initialize="Xavier",activation="sigmoid",learning_rate=0.05,update="MBGD",batchsize=32,lambd=0,keep_prob=None,beta_1=0.9,beta_2=0.999,epsilon=1e-8,batch_normalize=False,beta_bn=0.99):
        self.layer_num=len(layer_vec)
        self.layer_vec=layer_vec
        self.initialize=initialize
        self.activation=activation
        self.lr=learning_rate
        self.update=update
        self.batchsize=batchsize
        self.lambd=lambd     #lambd是用于L2正则化的超参数
        if keep_prob is None:
            self.keep_prob_back_up=np.ones_like(self.layer_vec,dtype=np.float32)
        else:
            if isinstance(keep_prob,np.ndarray):
                self.keep_prob_back_up=keep_prob
            else:
                self.keep_prob_back_up=np.array(keep_prob)
        #神经网络转到训练状态（开启BN和dropout）
        self.to_train()
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon
        self.batch_normalize=batch_normalize
        self.beta_bn=beta_bn
        if self.initialize=="Xavier":   #这里是Xavier均匀分布
            #各层的W,B矩阵,W[0]为None
            self.W=[None if i==0 else np.random.uniform( -np.sqrt(3/self.layer_vec[i-1]),np.sqrt(3/self.layer_vec[i-1]),(self.layer_vec[i],self.layer_vec[i-1]) ) for i in range(self.layer_num)]
            self.B=[None if i==0 else np.random.uniform( -1/np.sqrt(self.layer_vec[i-1]),1/np.sqrt(self.layer_vec[i-1]),(self.layer_vec[i],1) ) for i in range(self.layer_num)]
        elif self.initialize=="kaiming":
            self.W = [None if i == 0 else np.random.normal(0, 8, (self.layer_vec[i],self.layer_vec[i-1])) for i in range(self.layer_num)]
            self.B = [None if i == 0 else np.zeros((self.layer_vec[i],1),dtype=np.float32) for i in range(self.layer_num)]
        if update=="Adam":
            #用于adam算法
            self.vdw=[None if i==0 else np.zeros_like(self.W[i]) for i in range(self.layer_num)]
            self.sdw=[None if i==0 else np.zeros_like(self.W[i]) for i in range(self.layer_num)]
            self.vdb=[None if i==0 else np.zeros_like(self.B[i]) for i in range(self.layer_num)]
            self.sdb=[None if i==0 else np.zeros_like(self.B[i]) for i in range(self.layer_num)]
        #本网络实现的bn算法是：训练阶段的平均值和方差取当前batch的，而测试阶段的平均值和方差是通过过去各个batch的指数加权平均而来
        if self.batch_normalize==True:
            self.EZ_add=[None if i == 0 else np.zeros((self.layer_vec[i],1),dtype=np.float32) for i in range(self.layer_num)]
            self.VarZ_add=[None if i == 0 else np.zeros((self.layer_vec[i],1),dtype=np.float32) for i in range(self.layer_num)]
            self.EZ_nowBatch=[None if i == 0 else np.zeros((self.layer_vec[i],1),dtype=np.float32) for i in range(self.layer_num)]
            self.Var_nowBatch=[None if i == 0 else np.zeros((self.layer_vec[i],1),dtype=np.float32) for i in range(self.layer_num)]
        #记录迭代的序列，用于偏差修正
        self.iter_num=1

        # #各层的激活值a,激活前的z,尚不知道具体的值，只是占个位置
        # self.a = [np.zeros((self.layer_vec[i], self.batchsize), dtype=np.float32) for i in range(self.layer_num)]
        # self.z = [np.zeros((self.layer_vec[i], self.batchsize), dtype=np.float32) for i in range(self.layer_num)]

    def to_eval(self):
        self.keep_prob = np.ones_like(self.keep_prob)
        self.state = "eval"

    def to_train(self):
        self.keep_prob = self.keep_prob_back_up
        self.state = "train"
    def  forward_calculate(self,X,Y):
        a = [np.zeros((self.layer_vec[i], X.shape[1]), dtype=np.float32) for i in range(self.layer_num)]
        z = [None if i == 0 else np.zeros((self.layer_vec[i], X.shape[1]), dtype=np.float32) for i in
             range(self.layer_num)]
        self.drop = [np.random.rand(self.layer_vec[i], X.shape[1]) <= self.keep_prob[i] for i in range(self.layer_num)]

        a[0] = copy.deepcopy(X)
        a[0] = a[0] * self.drop[0]
        a[0] /= self.keep_prob[0]
        if self.batch_normalize == True:
            z_ = [None if i == 0 else np.zeros((self.layer_vec[i], X.shape[1]), dtype=np.float32) for i in
                  range(self.layer_num - 1)]
            for i in range(1, self.layer_num - 1):
                z[i] = np.dot(self.W[i], a[i - 1]) + self.B[i]
                if self.state == "train":
                    # 计算期望和方差:
                    self.EZ_nowBatch[i] = np.sum(z[i], axis=1, keepdims=True) / z[i].shape[1]
                    self.VarZ_nowBatch[i] = np.sum((z[i] - self.EZ_nowBatch[i]) ** 2, axis=1, keepdims=True) / \
                                            z[i].shape[1]
                    self.VarZ_add[i] = self.beta_bn * self.VarZ_add[i] + (1 - self.beta_bn) * self.VarZ_nowBatch[i]
                    self.EZ_add[i] = self.beta_bn * self.EZ_add[i] + (1 - self.beta_bn) * self.EZ_nowBatch[i]
                if self.state == "train":
                    z_[i] = self.gamma[i] * (
                                (z[i] - self.EZ_nowBatch[i]) / np.sqrt(self.VarZ_nowBatch[i] + self.epsilon_BN)) + \
                            self.beta[i]
                else:
                    # 偏差修正
                    theta_BN = 1 / (1 - self.beta_bn ** self.iter_num)

                    z_[i] = self.gamma[i] * ((z[i] - theta_BN * self.EZ_add[i]) / np.sqrt(
                        theta_BN * self.VarZ_add[i] + self.epsilon_BN)) + self.beta[i]
                a[i] = eval("F." + self.activation)(z_[i])
                a[i] = a[i] * self.drop[i]
                a[i] /= self.keep_prob[i]


        else:

            for i in range(1, self.layer_num - 1):
                z[i] = np.dot(self.W[i], a[i - 1]) + self.B[i]

                a[i] = eval("F." + self.activation)(z[i])

                a[i] = a[i] * self.drop[i]

                a[i] /= self.keep_prob[i]

        # 损失函数为softmax+交叉熵的形式
        z[self.layer_num - 1] = np.dot(self.W[self.layer_num - 1], a[self.layer_num - 2]) + self.B[
            self.layer_num - 1]

        a[self.layer_num - 1] = F.softmax(z[self.layer_num - 1])
        a[self.layer_num - 1] = a[self.layer_num - 1] * self.drop[self.layer_num - 1]
        a[self.layer_num - 1] /= self.keep_prob[self.layer_num - 1]


        #记录z的值
        self.z=z
        #记录a的值
        self.a=a



        Loss_temp=np.zeros((Y.shape[1],),dtype=np.float32)
        for i in range(len(a[self.layer_num-1])):
            Loss_temp+=-Y[i]*np.log(a[self.layer_num-1][i])
        Loss_p1=np.sum(Loss_temp)/len(Loss_temp)

        Loss_temp=0
        for m in range(1,self.layer_num):
            Loss_temp+=np.sum(self.W[m]**2)
        Loss_p2=self.lambd/(2*Y.shape[1])*Loss_temp

        #保存损失值
        self.Loss=Loss_p1+Loss_p2
    def backward_calculate(self,X,Y):
        sample_num = X.shape[1]
        dz = [None if i == 0 else np.zeros((self.layer_vec[i], X.shape[1]), dtype=np.float32) for i in
              range(self.layer_num)]
        dw = [None if i == 0 else np.zeros((self.layer_vec[i], self.layer_vec[i - 1]), dtype=np.float32) for i in
              range(self.layer_num)]
        db = [None if i == 0 else np.zeros((self.layer_vec[i], 1), dtype=np.float32) for i in range(self.layer_num)]

        dz[self.layer_num - 1] = self.a[self.layer_num - 1] - Y
        dz[self.layer_num - 1] *= self.drop[self.layer_num - 1]
        if self.batch_normalize == True:
            d_var_z = [None if i == 0 else np.zeros((self.layer_vec[i], 1), dtype=np.float32) for i in
                       range(self.layer_num - 1)]
            d_EZ = [None if i == 0 else np.zeros((self.layer_vec[i], 1), dtype=np.float32) for i in
                    range(self.layer_num - 1)]
            dz_ = [None if i == 0 else np.zeros((self.layer_vec[i], X.shape[1]), dtype=np.float32) for i in
                   range(self.layer_num - 1)]
            d_gamma = [None if i == 0 else np.zeros((self.layer_vec[i], 1), dtype=np.float32) for i in
                       range(self.layer_num - 1)]
            d_beta = [None if i == 0 else np.zeros((self.layer_vec[i], 1), dtype=np.float32) for i in
                      range(self.layer_num - 1)]

        # 计算中间梯度
        if self.batch_normalize == True:
            for i in range(self.layer_num - 2, 0, -1):
                dz_[i] = eval("F." + self.activation + "__")(self.z_[i]) * np.dot(self.W[i + 1].transpose(), dz[i + 1])
                dz_[i] *= self.drop[i]
                # dz[i]=dz_[i]*self.gamma[i]/np.sqrt(self.VarZ_nowBatch[i]+self.epsilon_BN)
                d_var_z[i] = np.sum(dz_[i] * self.gamma[i] * (-1 / 2) * (self.z[i] - self.EZ_nowBatch[i]) * (
                            self.VarZ_nowBatch[i] + self.epsilon_BN) ** (-3 / 2), axis=1, keepdims=True)
                d_EZ[i] = (np.sum(dz_[i] * (-self.gamma[i]) * (self.VarZ_nowBatch[i] + self.epsilon_BN) ** (-1 / 2),
                                  axis=1, keepdims=True)) + (
                                      d_var_z[i] * (-2 / sample_num) * np.sum(self.z_[i] - self.EZ_nowBatch[i], axis=1,
                                                                              keepdims=True))
                dz[i] = dz_[i] * self.gamma[i] * (self.VarZ_nowBatch[i] + self.epsilon_BN) ** (-1 / 2) + d_var_z[i] * (
                            2 / sample_num) * (self.z[i] - self.EZ_nowBatch[i]) + d_EZ[i] * (1 / sample_num)
                # print("调试1:",d_var_z[i].shape)
                # print("调试2：",d_EZ[i].shape)
                # print("调试3:",dz[i].shape)
        else:
            for i in range(self.layer_num - 2, 0, -1):
                dz[i] = eval("F." + self.activation + "__")(self.z[i]) * np.dot(self.W[i + 1].transpose(), dz[i + 1])
                dz[i] *= self.drop[i]

        # 计算用于梯度更新的梯度
        if self.batch_normalize == True:
            for i in range(self.layer_num - 2, 0, -1):
                # 计算d_gamma和d_beta
                temp = dz_[i] * (self.z[i] - self.EZ_nowBatch[i]) / np.sqrt(self.VarZ_nowBatch[i] + self.epsilon_BN)
                # assert temp.shape[0]==self.layer_vec[i] and temp.shape[1]==sample_num
                d_gamma[i] = 1 / sample_num * np.sum(temp, axis=1, keepdims=True)
                d_beta[i] = 1 / sample_num * np.sum(dz_[i], axis=1, keepdims=True)
        for i in range(self.layer_num - 1, 0, -1):
            # 计算dw和db
            dw[i] = 1 / sample_num * np.dot(dz[i], self.a[i - 1].transpose()) + self.lambd / Y.shape[1] * self.W[i]
            db[i] = 1 / sample_num * np.sum(dz[i], axis=1, keepdims=True)

        # 进行梯度更新
        if self.update == "MBGD":
            for i in range(self.layer_num - 1, 0, -1):
                self.W[i] -= self.lr * dw[i]
                self.B[i] -= self.lr * db[i]
            if self.batch_normalize == True:
                for i in range(self.layer_num - 2, 0, -1):
                    self.gamma[i] -= self.lr * d_gamma[i]
                    self.beta[i] -= self.lr * d_beta[i]
        elif self.update == "Adam":
            # 偏差修正
            theta = (1 - self.beta_2 ** self.iter_num) ** 0.5 / (1 - self.beta_1 ** self.iter_num)

            for i in range(self.layer_num - 1, 0, -1):
                # 计算vdw,sdw,vdb,sdb
                self.vdw[i] = self.beta_1 * self.vdw[i] + (1 - self.beta_1) * dw[i]
                self.vdb[i] = self.beta_1 * self.vdb[i] + (1 - self.beta_1) * db[i]

                self.sdw[i] = self.beta_2 * self.sdw[i] + (1 - self.beta_2) * dw[i] ** 2
                self.sdb[i] = self.beta_2 * self.sdb[i] + (1 - self.beta_2) * db[i] ** 2

                self.W[i] -= theta * self.lr * self.vdw[i] / np.sqrt(self.sdw[i] + self.epsilon)
                self.B[i] -= theta * self.lr * self.vdb[i] / np.sqrt(self.sdb[i] + self.epsilon)
            if self.batch_normalize:
                for i in range(self.layer_num - 2, 0, -1):
                    # 计算vd_gamma,sd_gamma,vd_beta,sd_beta
                    self.vd_gamma[i] = self.beta_1 * self.vd_gamma[i] + (1 - self.beta_1) * d_gamma[i]
                    self.vd_beta[i] = self.beta_1 * self.vd_beta[i] + (1 - self.beta_1) * d_beta[i]

                    self.sd_gamma[i] = self.beta_2 * self.sd_gamma[i] + (1 - self.beta_2) * d_gamma[i] ** 2
                    self.sd_beta[i] = self.beta_2 * self.sd_beta[i] + (1 - self.beta_2) * d_beta[i] ** 2

                    self.gamma[i] -= theta * self.lr * self.vd_gamma[i] / np.sqrt(self.sd_gamma[i] + self.epsilon)
                    self.beta[i] -= theta * self.lr * self.vd_beta[i] / np.sqrt(self.sd_beta[i] + self.epsilon)
            self.iter_num += 1

            # 保存w,b,gamma，beta的梯度用于梯度验证
            # theta_valid=list()

            # theta_valid.extend(self.W)
            # theta_valid.extend(self.B)
            # theta_valid.extend(self.gamma)
            # theta_valid.extend(self.beta)

            # #记录梯度用于梯度验证(除以batchsize过后的)
            # theta_valid_der = list()
            # theta_valid_der.extend(dw)
            # theta_valid_der.extend(db)
            # if self.batch_normalize:
            #     theta_valid_der.extend(d_gamma)
            #     theta_valid_der.extend(d_beta)
            # self.theta_valid_der=theta_valid_der

    def begin_training(self,X:np.ndarray,Y:np.ndarray,VX:np.ndarray,VY:np.ndarray,epochnum=3):
        #打乱操作
        X_Y=np.concatenate((X,Y),axis=0)
        X_Y_temp=X_Y.transpose()
        np.random.shuffle(X_Y_temp)
        X_Y=X_Y_temp.transpose()

        self.X=X_Y[:len(X)]
        self.Y=X_Y[len(X):]

        batch_num=self.X.shape[1]//self.batchsize

        val_batch_num=VX.shape[1]//self.batchsize

        for epoch in range(epochnum):
            for iter in range(batch_num):
                self.forward_calculate(self.X[:,iter*self.batchsize:(iter+1)*self.batchsize],self.Y[:,iter*self.batchsize:(iter+1)*self.batchsize])
                self.backward_calculate(self.X[:,iter*self.batchsize:(iter+1)*self.batchsize],self.Y[:,iter*self.batchsize:(iter+1)*self.batchsize])

            #每个epoch检查一下正确率
            cor_rate=0
            loss=0
            iter=None
            self.to_eval()
            for iter in range(val_batch_num):
                self.forward_calculate(VX[:,iter*self.batchsize:(iter+1)*self.batchsize],VY[:,iter*self.batchsize:(iter+1)*self.batchsize])
                cor_num = np.sum(np.argmax(self.a[self.layer_num-1],axis=0)==np.argmax(VY[:,iter*self.batchsize:(iter+1)*self.batchsize],axis=0))
                now_cor_rate = cor_num/self.batchsize
                cor_rate=now_cor_rate/(iter+1)+iter/(iter+1)*cor_rate
                loss=self.Loss/(iter+1)+iter/(iter+1)*loss
            print("epoch{},正确率为{},平均损失为{}".format(epoch,cor_rate,loss))
            self.to_train()
            # print("iter:",iter)
            # 打印结果便于查看效果
            # print(self.a[self.layer_num-1])
            # print(VY[:,iter*self.batchsize:(iter+1)*self.batchsize])
            if cor_rate>0.9995:
                print("准确率已经达到要求，当前为{:.2f}%，正在早停。。。".format(cor_rate*100))
                break

    def valify_der(self):
        print("开始梯度检测")
        self.to_train()
        x_t = np.array([[1], [2]], dtype=np.float32)
        y_t = np.array([[0], [1]], dtype=np.float32)
        sample_num = x_t.shape[1]
        self.forward_calculate(x_t, y_t)
        self.backward_calculate(x_t, y_t)
        theta_valid_der = self.theta_valid_der

        theta_valid = list()
        theta_valid.extend(self.W)
        theta_valid.extend(self.B)
        if self.batch_normalize:
            theta_valid.extend(self.gamma)
            theta_valid.extend(self.beta)
        vec1=list()
        vec2=list()

        for layer in range(len(theta_valid)):
            if theta_valid[layer] is not None:
                for idx1 in range(len(theta_valid[layer])):
                    for idx2 in range(len(theta_valid[layer][idx1])):
                        eps = 1e-7
                        delta = 1e-7
                        back_up = theta_valid[layer][idx1][idx2]

                        theta_valid[layer][idx1][idx2] = back_up - eps
                        self.forward_calculate(x_t, y_t)
                        loss1 = self.Loss * sample_num

                        theta_valid[layer][idx1][idx2] = back_up + eps
                        self.forward_calculate(x_t, y_t)
                        loss2 = self.Loss * sample_num

                        ahp_temp=(loss2-loss1)/(2*eps)
                        bet_temp=theta_valid_der[layer][idx1][idx2]
                        vec1.append(ahp_temp)
                        vec2.append(bet_temp)

                        # 复位
                        theta_valid[layer][idx1][idx2] = back_up

                        # if 1e-4>check>1e-6:
                        #     print("warning!")
                        #     print(layer,idx1,idx2)
                        # elif check>=1e-4:
                        #     print("error!")
                        #     print("evaluate:",d_eval)
                        #     print("real:",theta_valid_der[layer][idx1][idx2])
                        #     print("error!")
                        #     print("location:",layer, idx1, idx2)
                        #     print("check value:",check)
        vec1=np.array(vec1)
        vec2=np.array(vec2)
        f1=vec1-vec2
        check=np.sqrt(np.sum(f1**2))/np.sqrt(np.sum(vec1**2))+np.sqrt(np.sum(vec2**2))
        if 1e-5>check>1e-7:
            print("warning!")
            print("check value:",check)
        elif check>=1e-5:
            print("error!")
            print("check value:",check)
        print("结束梯度检测")

if __name__=="__main__":
    # 随便准备了点数据，用于看看效果。
    # 画了个圆，正方形内均匀随机生成5万个点，圆内的点为正样本，圆外的点为负样本
    x = np.zeros((50000, 2), float)
    y = np.zeros((50000, 2), float)
    for i in range(len(x)):
        x[i] = np.random.uniform(0,6,(2,))
        if (x[i][0] - 3) ** 2 + (x[i][1] - 3) ** 2 <= 4:
            y[i][0] = 1
        else:
            y[i][1] = 1
    # 对输入数据作归一化
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0, ddof=1)  # ddof=1说明是样本标准差（除以n-1）
    for i in range(len(x)):
        x[i] = (x[i] - x_mean) / x_std

    x=x.transpose()
    y=y.transpose()


    # def __init__(self, layer_vec, *, initialize="Xavier", activation="sigmoid", learning_rate=0.05, update="MBGD",
    #              batchsize=32, lambd=0, keep_prob=None, beta_1=0.9, beta_2=0.999):
    NN = MyNeuralNetwork([2, 200, 100, 2],keep_prob=[1,1,1,1],learning_rate=0.001,update="Adam")
    NN.begin_training(x[:,:40000],y[:,:40000],x[:,40000:],y[:,40000:])
    NN.valify_der()








