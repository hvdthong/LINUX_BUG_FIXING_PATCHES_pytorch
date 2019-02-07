import torch.nn as nn
import torch
import torch.nn.functional as F


class PatchNet(nn.Module):
    def __init__(self, args):
        super(PatchNet, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-3D for commit code
        code_line = args.code_line  # the number of LOC in each hunk of commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_hunk = nn.ModuleList([nn.Conv3d(Ci, Co, (K, code_line, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(1 * len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        # ver_2: get the vector of hunk, apply cnn2d over the hunks
        x = torch.mean(x, dim=3)  # get the line vector
        x = torch.mean(x, dim=2)  # get the hunk vector
        x = self.forward_msg(x=x, convs=convs_line)
        return x

        # ver 1: apply cnn-2d over a list of words
        # n_batch, n_hunk, n_line = x.shape[0], x.shape[1], x.shape[2]
        # x = x.reshape(n_batch * n_hunk * n_line, x.shape[3], x.shape[4])
        #
        # # apply cnn 2d for each line in a commit code
        # x = self.forward_msg(x=x, convs=convs_line)
        #
        # # apply mean for each line, each hunk
        # x = x.reshape(n_batch, n_hunk, n_line, self.args.num_filters * len(self.args.filter_sizes))
        # x = torch.mean(x, dim=2)  # get the hunk vector
        # x = torch.mean(x, dim=1)  # get the file vector
        # return x

    def forward(self, msg, added_code, removed_code):
        # x_msg = self.embed_msg(msg.cuda() if torch.cuda.is_available() else msg)
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        # # x_added_code = self.embed_code(added_code.cuda() if torch.cuda.is_available() else added_code)
        # x_added_code = self.embed_code(added_code)
        # x_added_code = self.forward_code(x_added_code, self.convs_code_line, self.convs_code_hunk)
        #
        # # x_removed_code = self.embed_code(removed_code.cuda() if torch.cuda.is_available() else removed_code)
        # x_removed_code = self.embed_code(removed_code)
        # x_removed_code = self.forward_code(x_removed_code, self.convs_code_line, self.convs_code_hunk)
        #
        # # x_diff = x_added_code - x_removed_code  # measure the diff of the code changes
        #
        # x_commit = torch.cat((x_msg, x_added_code, x_removed_code), 1)

        x_commit = x_msg
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

    def forward_commit_embeds(self, msg, added_code, removed_code):
        x_msg = self.embed_msg(msg.cuda() if torch.cuda.is_available() else msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_added_code = self.embed_code(added_code.cuda() if torch.cuda.is_available() else added_code)
        x_added_code = self.forward_code(x_added_code, self.convs_code_line, self.convs_code_hunk)
        x_removed_code = self.embed_code(removed_code.cuda() if torch.cuda.is_available() else removed_code)
        x_removed_code = self.forward_code(x_removed_code, self.convs_code_line, self.convs_code_hunk)

        x_diff = x_added_code - x_removed_code  # measure the diff of the code changes

        x_commit = torch.cat((x_msg, x_diff), 1)
        return x_commit