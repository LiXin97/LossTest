mport numpy as np
import matplotlib.pyplot as plt
import random
import math

class Loss:
    def __init__(self, thread):
        self._thread = thread
        self._thread2 = thread * thread
        self. _losswight = 1e16

    def renewloss(self):
        self._losswight = 1.

    def getthread(self):
        return self._thread

    def getlossw(self):
        return self._losswight

class TrivialLoss(Loss):
    name = "TrivialLoss"

    def __init__(self, thread):
        super().__init__(thread)

    def updatelossoutlier(self, error):
        self._losswight = 1.

    def updatelossinlier(self, error):
        self._losswight = 1.

class HuberLoss(Loss):
    name = "HuberLoss"

    def __init__(self, thread):
        super().__init__(thread)

    def updatelossoutlier(self, error):
        self._losswight = max( 1e-6, self._thread / error )

    def updatelossinlier(self, error):
        self._losswight = 1.

class SoftLOneLoss(Loss):
    name = "SoftLOneLoss"

    def __init__(self, thread):
        super().__init__(thread)

    def updatelossoutlier(self, error):
        self.updatelossinlier(error)

    def updatelossinlier(self, error):
        sum = 1. + error*error / self._thread2
        self._losswight = max( 1e-16, 1. / math.sqrt(sum) )

class CauchyLoss(Loss):
    name = "CauchyLoss"

    def __init__(self, thread):
        super().__init__(thread)

    def updatelossoutlier(self, error):
        self.updatelossinlier(error)

    def updatelossinlier(self, error):
        sum = 1. + error*error / self._thread2
        self._losswight = max( 1e-16, 1. / sum )

class TukeyLoss(Loss):
    name = "TukeyLoss"

    def __init__(self, thread):
        super().__init__(thread)

    def updatelossoutlier(self, error):
        self._losswight = 0

    def updatelossinlier(self, error):
        value = 1. - (error * error) / self._thread2
        self._losswight = .5 * ( value * value)

class LossSimu:
    true_paramter = [ 1., 2., 1.]
    result0 = [0.,0.]
    result1 = [0.,0.]
    result2 = [0.,0.]
    result3 = [0.,0.]
    result4 = [0.,0.]

    def __init__(self, nums_data, outliers, loops):
        self._nums_data = nums_data
        self._outliers = outliers
        self._loops = loops

    def add_guass_noise(self, observe_y, mean, sigma):
        noise = np.random.normal(mean, sigma, len(observe_y))
        for index in range(len(observe_y)):
            observe_y[index] += noise[index]#random.gauss(mean, sigma)

    def add_outlier_noise(self, observe_y, min, max):
        for i in range(self._outliers):
            observe_y[random.sample(range(0, len(observe_y)), 1)] += random.uniform(min, max)

    def check_result(self, parameter_out, it_time_out, all_result):
        all_result[0] += (parameter_out[0] - self.true_paramter[0]) * (parameter_out[0] - self.true_paramter[0]) \
                         + (parameter_out[1] - self.true_paramter[1]) * (parameter_out[1] - self.true_paramter[1]) \
                         + (parameter_out[2] - self.true_paramter[2]) * (parameter_out[2] - self.true_paramter[2])
        all_result[1] += it_time_out

    def GN_Loss(self, parameter, observe_x, observe_y, loss):
        iters = 10000
        opti_a = parameter[0]
        opti_b = parameter[1]
        opti_c = parameter[2]

        iter_time = 0

        for i in range(iters):
            inlier_num = 0
            outlier_num = 0
            hessen = np.zeros((3, 3))
            error_b = np.zeros((3, 1))
            delta = np.zeros((3, 1))

            for index in range(len(observe_x)):
                jacobian = np.zeros((3, 1))
                pred = math.exp(opti_a * observe_x[index] * observe_x[index] + opti_b * observe_x[index] + opti_c)
                error = pred - observe_y[index]
                if abs(error) > loss.getthread():
                    loss.updatelossoutlier(error)
                    outlier_num += 1
                else:
                    loss.updatelossinlier(error)
                    inlier_num += 1
                jacobian[0] = observe_x[index] * observe_x[index] * pred
                jacobian[1] = observe_x[index] * pred
                jacobian[2] = pred
                jacobianT = jacobian.transpose()
                hessen_now = loss.getlossw() * jacobian * jacobian.transpose()
                hessen = hessen + hessen_now
                error_b -= loss.getlossw() * jacobian * error
            hessen_inv = np.ones((3,3))
            try:
                hessen_inv = np.linalg.inv(hessen)
            except:
                print(hessen_inv)
            delta = np.matmul(hessen_inv, error_b)
            opti_a += delta[0]
            opti_b += delta[1]
            opti_c += delta[2]

            print(inlier_num / len(observe_x))

            if delta.transpose().dot(delta) < 1e-9:
                print("delta.sum() ", loss.name, np.sum(delta**2), " iter time = ", i)
                iter_time = i
                break
        return [opti_a, opti_b, opti_c], iter_time

    def show(self, observe_x, observe_y, paramter0, paramter1, paramter2, paramter3, paramter4):
        plt.axis([0,observe_x.max(),0,observe_y.max()])
        plt.plot(observe_x, observe_y, 'ro')
        observe_y = np.exp(paramter0[0] * observe_x * observe_x + paramter0[1] * observe_x + paramter0[2])
        plt.plot(observe_x, observe_y, '--')
        observe_y = np.exp(paramter1[0] * observe_x * observe_x + paramter1[1] * observe_x + paramter1[2])
        plt.plot(observe_x, observe_y, '-')
        observe_y = np.exp(paramter2[0] * observe_x * observe_x + paramter2[1] * observe_x + paramter2[2])
        plt.plot(observe_x, observe_y, '-.')
        observe_y = np.exp(paramter3[0] * observe_x * observe_x + paramter3[1] * observe_x + paramter3[2])
        plt.plot(observe_x, observe_y, ':')
        observe_y = np.exp(paramter4[0] * observe_x * observe_x + paramter4[1] * observe_x + paramter4[2])
        plt.plot(observe_x, observe_y, '+')
        plt.show()

    def run(self):
        for loop in range(self._loops):
            observe_x = np.linspace(0, 1., self._nums_data)
            observe_y = np.exp(self.true_paramter[0]*observe_x*observe_x + self.true_paramter[1]*observe_x + self.true_paramter[2])
            # for index in range(len(observe_x)):
            #     pred = math.exp(self.true_paramter[0] * observe_x[index] * observe_x[index] + self.true_paramter[1] * observe_x[index] + self.true_paramter[2])
            #     error = pred - observe_y[index]
            #     print( error )
            self.add_guass_noise(observe_y, 0., .5)
            self.add_outlier_noise(observe_y, -20, 20.)
            # mean = 0.
            # sqrt = 0.
            # for index in range(len(observe_x)):
            #     pred = math.exp(self.true_paramter[0] * observe_x[index] * observe_x[index] + self.true_paramter[1] * observe_x[index] + self.true_paramter[2])
            #     error = pred - observe_y[index]
            #     mean += error
            #     sqrt += error * error
            #     print( error )
            # print(mean / len(observe_x))
            # print(sqrt / len(observe_x))
            # parameter_give = [.95, 1.8, .8]
            parameter_give = [1., 2., 1.]
            thread = 1.
            # print("before opti", parameter_give)
            trivialLoss = TrivialLoss(thread)
            parameter0, times = self.GN_Loss(parameter_give, observe_x, observe_y, trivialLoss)
            self.check_result(parameter0, times, self.result0)
            # print("after trivialLoss opti", parameter)

            huberloss = HuberLoss(thread)
            parameter1, times = self.GN_Loss(parameter_give, observe_x, observe_y, huberloss)
            self.check_result(parameter1, times, self.result1)
            # print("after huber opti", parameter)

            softLOneLoss = SoftLOneLoss(thread)
            parameter2, times = self.GN_Loss(parameter_give, observe_x, observe_y, softLOneLoss)
            self.check_result(parameter2, times, self.result2)
            # print("after SoftLOneLoss opti", parameter)

            cauchyLoss = CauchyLoss(thread)
            parameter3, times = self.GN_Loss(parameter_give, observe_x, observe_y, cauchyLoss)
            self.check_result(parameter3, times, self.result3)
            # print("after CauchyLoss opti", parameter)

            tukeyloss = TukeyLoss(thread)
            parameter4, times = self.GN_Loss(parameter_give, observe_x, observe_y, tukeyloss)
            self.check_result(parameter4, times, self.result4)
            # print("after tukeyloss opti", parameter)

            self.show(observe_x, observe_y, parameter0, parameter1, parameter2, parameter3, parameter4)

    def output(self):
        print("trivialLoss opti result ", self.result0[0]/self._loops, "trivialLoss opti avager time ", self.result0[1]/self._loops)
        print("huber opti result ", self.result1[0]/self._loops, "huber opti avager time ", self.result1[1]/self._loops)
        print("SoftLOneLoss opti result ", self.result2[0]/self._loops, "SoftLOneLoss opti avager time ", self.result2[1]/self._loops)
        print("CauchyLoss opti result ", self.result3[0]/self._loops, "CauchyLoss opti avager time ", self.result3[1]/self._loops)
        print("tukeyloss opti result ", self.result4[0]/self._loops, "tukeyloss opti avager time ", self.result4[1]/self._loops)


losssimu = LossSimu(50, 5, 1)
losssimu.run()
losssimu.output()
