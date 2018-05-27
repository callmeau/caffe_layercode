import numpy as np
class Forword():
    def __init__(self,startI,tranI,tranO):
        self.startI = startI
        self.tranO = tranO
        self.tranI = tranI
    def init_a(self,_cur):
        ac = self.startI * self.tranO[:,_cur]
        print ac
        return ac
    def calculate_next(self,ac,_cur):
        tmp = ac[:,np.newaxis] * self.tranI
        pro = self.tranO[:,_cur]
        ac_next = np.sum(tmp,axis = 0) * pro
        return ac_next
    def run(self,seq):
        T = len(seq)
        ac = self.init_a(seq[0])
        for t in range(1,T):
            _cur = seq[t]
            ac = self.calculate_next(ac,_cur)
            print ac
        return np.sum(ac)

if __name__ == '__main__':
    A = np.array([0.5,0.2, .3,.3 , .5, .2, .2, .3, .5])
    A = A.reshape(3,3)
    print '=' * 10,'transpose mat'
    print A
    B = np.array([.5, .5, .4, .6, .7,.3])
    B = B.reshape(3,2)
    print '=' * 10,'train statu  mat'
    print B
    pi = np.array([.2, .4, .4])
    model = Forword(pi,A,B)
    seq = (0,1,0)
    ac = model.run(seq)
    print ac

            


        
