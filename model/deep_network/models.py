import torch.nn as nn
import torch.nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,dim_input, dim_hidden,dropout_p = 0.0):
        super(MLP,self).__init__()
        p = dropout_p

        self.linear1 = torch.nn.Linear(dim_input, dim_hidden)
        self.relu1 = torch.nn.LeakyReLU()
        self.bn1 = torch.nn.BatchNorm1d(dim_hidden)
        self.dropout1 = torch.nn.Dropout(p)

        self.linear2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu2 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.BatchNorm1d(dim_hidden)
        self.dropout2 = torch.nn.Dropout(p)

        self.linear3 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu3 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.BatchNorm1d(dim_hidden)
        self.dropout3 = torch.nn.Dropout(p)

        self.linear4 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu4 = torch.nn.LeakyReLU()
        self.bn4 = torch.nn.BatchNorm1d(dim_hidden)
        self.dropout4 = torch.nn.Dropout(p)

        self.linear5 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.relu5 = torch.nn.LeakyReLU()
        self.bn5 = torch.nn.BatchNorm1d(dim_hidden)
        self.dropout5 = torch.nn.Dropout(p)

        self.linear6 = torch.nn.Linear(dim_hidden, dim_hidden // 2)
        self.relu6 = torch.nn.LeakyReLU()
        self.bn6 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout6 = torch.nn.Dropout(p)

        self.linear7 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu7 = torch.nn.LeakyReLU()
        self.bn7 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout7 = torch.nn.Dropout(p)

        self.linear8 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu8 = torch.nn.LeakyReLU()
        self.bn8 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout8 = torch.nn.Dropout(p)

        self.linear9 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu9 = torch.nn.LeakyReLU()
        self.bn9 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout9 = torch.nn.Dropout(p)

        self.linear10 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu10 = torch.nn.LeakyReLU()
        self.bn10 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout10 = torch.nn.Dropout(p)

        self.linear11 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu11 = torch.nn.LeakyReLU()
        self.bn11 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout11 = torch.nn.Dropout(p)

        self.linear12 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu12 = torch.nn.LeakyReLU()
        self.bn12 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout12 = torch.nn.Dropout(p)

        self.linear13 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu13 = torch.nn.LeakyReLU()
        self.bn13 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout13 = torch.nn.Dropout(p)

        self.linear14 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu14 = torch.nn.LeakyReLU()
        self.bn14 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout14 = torch.nn.Dropout(p)

        self.linear15 = torch.nn.Linear(dim_hidden // 2, dim_hidden // 2)
        self.relu15 = torch.nn.LeakyReLU()
        self.bn15 = torch.nn.BatchNorm1d(dim_hidden// 2)
        self.dropout15 = torch.nn.Dropout(p)

        self.linear_out = torch.nn.Linear(dim_hidden // 2, 1)
    def forward(self,vX):
        lin1 = self.linear1(vX)
        h_relu1 = self.relu1(lin1)
        h_relu1 = self.bn1(h_relu1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        h_relu2 = self.relu2(lin2)
        h_relu2 = self.bn2(h_relu2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        h_relu3 = self.relu3(lin3)
        h_relu3 = self.bn3(h_relu3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        h_relu4 = self.relu4(lin4)
        h_relu4 = self.bn4(h_relu4)
        drop4 = self.dropout4(h_relu4)

        lin5 = self.linear5(drop4)
        h_relu5 = self.relu5(lin5)
        h_relu5 = self.bn5(h_relu5)
        drop5 = self.dropout5(h_relu5)

        lin6 = self.linear6(drop5)
        h_relu6 = self.relu6(lin6)
        h_relu6 = self.bn6(h_relu6)
        drop6 = self.dropout6(h_relu6)

        lin7 = self.linear7(drop6)
        h_relu7 = self.relu7(lin7)
        h_relu7 = self.bn7(h_relu7)
        drop7 = self.dropout7(h_relu7)

        lin8 = self.linear8(drop7)
        h_relu8 = self.relu8(lin8)
        h_relu8 = self.bn8(h_relu8)
        drop8 = self.dropout8(h_relu8)

        lin9 = self.linear9(drop8)
        h_relu9 = self.relu9(lin9)
        h_relu9 = self.bn9(h_relu9)
        drop9 = self.dropout9(h_relu9)

        lin10 = self.linear10(drop9)
        h_relu10 = self.relu10(lin10)
        h_relu10 = self.bn10(h_relu10)
        drop10 = self.dropout10(h_relu10)

        lin11 = self.linear11(drop10)
        h_relu11 = self.relu11(lin11)
        h_relu11 = self.bn11(h_relu11)
        drop11 = self.dropout11(h_relu11)

        lin12 = self.linear12(drop11)
        h_relu12 = self.relu12(lin12)
        h_relu12 = self.bn12(h_relu12)
        drop12 = self.dropout12(h_relu12)

        lin13 = self.linear13(drop12)
        h_relu13 = self.relu13(lin13)
        h_relu13 = self.bn13(h_relu13)
        drop13 = self.dropout13(h_relu13)

        lin14 = self.linear14(drop13)
        h_relu14 = self.relu14(lin14)
        h_relu14 = self.bn14(h_relu14)
        drop14 = self.dropout14(h_relu14)

        lin15 = self.linear15(drop14)
        h_relu15 = self.relu15(lin15)
        h_relu15 = self.bn15(h_relu15)
        drop15 = self.dropout15(h_relu15)

        out = self.linear_out(drop15)

        return out

    def get_loss(self,weights,labels):
        predict_loss = F.mse_loss(weights,labels)
        return predict_loss

    def output_weights(self,writter,epoch):
        writter.add_histogram('forward_weight1/weights',self.linear1.weight,epoch+1)
        writter.add_histogram('forward_weight6/weights',self.linear6.weight,epoch+1)
        writter.add_histogram('forward_weight11/weights',self.linear11.weight,epoch+1)
        writter.add_histogram('forward_weight15/weights',self.linear15.weight,epoch+1)


