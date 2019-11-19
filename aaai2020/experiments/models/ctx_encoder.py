"""
Set of context encoders.
"""
from itertools import combinations
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

from models.utils import *


class AttributeContextEncoder(nn.Module):
    def __init__(self, domain, args):
        super(AttributeContextEncoder, self).__init__()

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        assert args.nembed_ctx % self.dim_ent == 0

        self.hidden_size = int(args.nembed_ctx / self.dim_ent)
        #self.hidden_size = int(args.nembed_ctx / 2)

        self.x_value_encoder = nn.Sequential(
            torch.nn.Linear(domain.num_ent(), self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.y_value_encoder = nn.Sequential(
            torch.nn.Linear(domain.num_ent(), self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.size_encoder = nn.Sequential(
            torch.nn.Linear(domain.num_ent(), self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.color_encoder = nn.Sequential(
            torch.nn.Linear(domain.num_ent(), self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        init_cont([self.color_encoder, self.size_encoder, self.x_value_encoder, self.y_value_encoder], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        ents = ctx_t.view(ctx_t.size(0), self.num_ent, self.dim_ent, 1)

        all_relative_ents = []
        for i in range(self.num_ent):
            relative_ents = []
            for j in range(self.num_ent):
                if i == j:
                    relative_ent = ents[:,i,:,:]
                else:
                    relative_ent = ents[:,i,:,:] - ents[:,j,:,:]
                relative_ents.append(relative_ent.unsqueeze(1))
            all_relative_ents.append(torch.cat(relative_ents, 3))
        all_relative_ents = torch.cat(all_relative_ents, 1)

        x_values = self.x_value_encoder(all_relative_ents[:,:,0,:])
        y_values = self.y_value_encoder(all_relative_ents[:,:,1,:])
        #sizes =  self.size_encoder(all_relative_ents[:,:,2,:])
        #colors = self.color_encoder(all_relative_ents[:,:,3,:])

        #out = torch.cat([x_values, y_values, sizes, colors], 2)
        out = torch.cat([x_values, y_values], 2)
        #out = torch.cat([sizes, colors], 2)

        return out


class AttentionContextEncoder(nn.Module):
    def __init__(self, domain, args):
        super(AttentionContextEncoder, self).__init__()

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.property_encoder = nn.Sequential(
            #nn.Dropout(args.dropout),
            #torch.nn.Linear(domain.dim_ent(), args.nembed_ctx),
            torch.nn.Linear(domain.dim_ent(), int(args.nembed_ctx / 2)),
            #torch.nn.Linear(domain.dim_ent(), int(args.nembed_prop)),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Dropout(args.dropout)
        )

        self.relation_encoder = nn.Sequential(
            #nn.Dropout(args.dropout),
            #torch.nn.Linear(2 * domain.dim_ent(), args.nembed_ctx),
            #torch.nn.Linear(2 * domain.dim_ent(), int(args.nembed_rel)),
            #torch.nn.Linear(domain.dim_ent(), int(args.nembed_rel)),
            #torch.nn.Linear(domain.dim_ent() + 1, int(args.nembed_rel)),
            torch.nn.Linear(domain.dim_ent() + 1, int(args.nembed_ctx / 2)),
            nn.Tanh(),
            #nn.ReLU(),
            nn.Dropout(args.dropout)
        )

        self.dropout = nn.Dropout(args.dropout)
        #self.fc1 = nn.Linear(args.nembed_ctx, args.nembed_ctx) 
        #self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()

        init_cont([self.property_encoder, self.relation_encoder], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        ents = ctx_t.view(ctx_t.size(0), self.num_ent, self.dim_ent)
        prop_emb = self.property_encoder(ents)
        ent_rel_pairs = []
        for i in range(self.num_ent):
            rel_pairs = []
            for j in range(self.num_ent):
                if i == j:
                    continue
                #rel_pairs.append(torch.cat([ents[:,i,:],ents[:,j,:]], 1).unsqueeze(1))
                #rel_pairs.append((ents[:,i,:] - ents[:,j,:]).unsqueeze(1))
                dist = torch.sqrt((ents[:,i,0] - ents[:,j,0])**2 + (ents[:,i,1] - ents[:,j,1])**2)
                rel_pairs.append((torch.cat([ents[:,i,:] - ents[:,j,:], dist.unsqueeze(1)], 1).unsqueeze(1)))
                #rel_pairs.append((torch.cat([ents[:,i,:] - ents[:,j,:], (ents[:,i,:] - ents[:,j,:])**2], 1).unsqueeze(1)))
            ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
        ent_rel_pairs = torch.cat(ent_rel_pairs, 1)
        rel_emb = self.relation_encoder(ent_rel_pairs).sum(2)
        #rel_emb = self.relation_encoder(ent_rel_pairs).max(2)[0]
        #rel_emb = self.relation_encoder(ent_rel_pairs)
        #rel_emb = rel_emb.view(rel_emb.size(0), rel_emb.size(1), -1)
        #rel_emb = ent_rel_pairs.view(ent_rel_pairs.size(0), ent_rel_pairs.size(1), -1)
        out = torch.cat([prop_emb, rel_emb], 2)
        #out = torch.add(prop_emb, rel_emb)
        #out = self.fc1(out)
        #out = self.tanh(out)
        #out = self.relu(out)
        #out = self.dropout(out)
        return out



class RelationContextEncoder(nn.Module):
    def __init__(self, domain, args):
        super(RelationContextEncoder, self).__init__()

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()
        num_rel = int(domain.num_ent() * (domain.num_ent() - 1) / 2)

        self.relation_encoder = nn.Sequential(
            torch.nn.Linear(2 * domain.dim_ent(), args.nhid_rel),
            nn.Tanh()
        )

        self.fc1 = nn.Linear(num_rel * args.nhid_rel, args.nembed_ctx) 
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(args.dropout)

        init_cont([self.relation_encoder, self.fc1], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        ents = ctx_t.view(ctx_t.size(0), self.num_ent, self.dim_ent)

        rel_pairs = []
        for i in range(self.num_ent):
            for j in range(self.num_ent):
                if i < j:
                    rel_pairs.append(torch.cat([ents[:,i,:],ents[:,j,:]], 1).unsqueeze(1))
        rel_pairs = torch.cat(rel_pairs, 1)        
        out = self.relation_encoder(rel_pairs).view(rel_pairs.size(0), -1)
        out = self.fc1(out)
        out = self.tanh(out)
        out = self.dropout(out)
        return out


class MlpContextEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, domain, args):
        super(MlpContextEncoder, self).__init__()

        self.fc1 = nn.Linear(domain.num_ent() * domain.dim_ent(), args.nembed_ctx)
        self.fc2 = nn.Linear(args.nembed_ctx, args.nembed_ctx)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(args.dropout)

        init_cont([self.fc1, self.fc2], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        out = self.fc1(ctx_t)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.dropout(out)
        return out

