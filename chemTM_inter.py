

import numpy as np
import random

from PIL import Image
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, DataStructs, rdFingerprintGenerator
from rdkit.Chem import Draw, rdDepictor

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FormatStrFormatter

from io import BytesIO
from collections import Counter
from rdkit.Chem.Draw import SimilarityMaps

from scipy.spatial.distance import cdist


def show_mol(d2d):
    bio = BytesIO(d2d.GetDrawingText())
    return Image.open(bio)

def show_images(imgs, nCols,buffer=5):
    max_height = 0
    max_width = 0
    for img in imgs:
        max_height = max(max_height,img.height)
        max_width = max(max_width,img.width)
    width = max_width * nCols
    nRows = int(np.ceil(len(imgs)/nCols))
    height = max_height * nRows
    width += buffer*(len(imgs)-1)
    res = Image.new("RGBA",(width,height))
    x = 0
    col= 0
    row=0
    for img in imgs:
        if col >= nCols:
            col = 0
            row += 1
            x = 0
        y = row * max_height + row*buffer
        res.paste(img,(x,y))
        x += img.width + buffer
        col += 1
    return res

def fp_to_np(fp):
    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def includeRingMembership(s, n):
    r=';R]'
    d="]"
    return r.join([d.join(s.split(d)[:n]), d.join(s.split(d)[n:])])

def includeDegree(s, n, d):
    r=';D'+str(d)+']'
    d="]"
    return r.join([d.join(s.split(d)[:n]), d.join(s.split(d)[n:])])

def writePropsToSmiles(mol, smi, order):
    finalsmi = smi
    for i,a in enumerate(order):
        atom = mol.GetAtomWithIdx(a)
        if atom.IsInRing():
            finalsmi = includeRingMembership(finalsmi, i+1)
        finalsmi = includeDegree(finalsmi, i+1, atom.GetDegree())
    return finalsmi

def getSubstructureSmi(mol, atomID, radius):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomID)
        atomsToUse = []
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    smi = Chem.MolFragmentToSmiles(mol, atomsToUse, bondsToUse=env, allHsExplicit=True, allBondsExplicit=True, rootedAtAtom=atomID)
    order = eval(mol.GetProp("_smilesAtomOutputOrder"))
    smi2 = writePropsToSmiles(mol, smi, order)
    return smi, smi2

def draw_clause(tm, W, posSMARTS, posSMARTS_indx, negSMARTS, negSMARTS_indx, X_train, training_smiles, n_neg_mols =2, bits_per_row = 5, panelSize=250, fpg=None):
    # clause_mols = [None] * n_cols
    # clause_logic = [str(active_W), str(inactive_W)] + [''] * (n_cols - 2)
    # need to store (example_mol, feature_index, ao.GetBitInfoMap()) for each literal
    # need to store separate list for legends
    # then draw 
    
    posMatch_indx = np.where(X_train[:, posSMARTS_indx].all(axis=1))[0]
    negMatch_indx = np.where(X_train[:, negSMARTS_indx].all(axis=1, where=0))[0]
    match_mols_indx = list(set.intersection(*map(set, [list(posMatch_indx), list(negMatch_indx)])))
    
    print(match_mols_indx)
    r_mol_index = random.randint(0, len(match_mols_indx)-1)
    match_smiles = training_smiles[match_mols_indx[r_mol_index]]
    match_mol = Chem.MolFromSmiles(match_smiles)
    rdDepictor.Compute2DCoords(match_mol)
    rdDepictor.StraightenDepiction(match_mol)
    match_atom_Ws = np.zeros(Descriptors.HeavyAtomCount(match_mol))
    

    mol_imgs = []
    clause_bits = []
    clause_logic = []
    if len(posSMARTS) > 0 :
        for smarts_idx in range(len(posSMARTS)):
            smarts = posSMARTS[smarts_idx]
            match_index = get_SMARTS_match_index(smarts, training_smiles)
            
            while True:
                r_index = random.randint(0, len(match_index)-1)
                if (X_train[match_index[r_index], posSMARTS_indx[smarts_idx]] == 1) and (Chem.MolFromSmiles(training_smiles[match_index[r_index]]).HasSubstructMatch(Chem.MolFromSmarts(smarts))):
                    break
                
            example_mol = Chem.MolFromSmiles(training_smiles[match_index[r_index]])
            rdDepictor.Compute2DCoords(example_mol)
            rdDepictor.StraightenDepiction(example_mol)
            
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fpg.GetFingerprint(example_mol, additionalOutput=ao)
            clause_bits.append((example_mol, posSMARTS_indx[smarts_idx], ao.GetBitInfoMap()))
            
            if smarts_idx == 0:
                clause_logic.append('')
            else:
                clause_logic.append('AND')
            
            # visualise compound
            patt = Chem.MolFromSmarts(smarts)
            matches = match_mol.GetSubstructMatches(patt)
            for match in matches:
                for atom_idx in match:
                    match_atom_Ws[atom_idx] += W / tm.number_of_clauses / len(matches)
            
        d2d = Draw.MolDraw2DCairo(550,350)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol=match_mol,
            weights=list(match_atom_Ws),
            draw2d=d2d,
            alpha=0,
            contourLines=10
        )
        
        d2d.FinishDrawing()
        mol_imgs.append(
            show_mol(d2d=d2d)
        )
            
            
    if len(negSMARTS) > 0:
        negStructure_vector = np.zeros(X_train.shape[1], dtype=np.int8)
        negStructure_vector[negSMARTS_indx] = 1
        
        neg_overlap = np.sum(negStructure_vector * X_train, axis=1)
        top_neg_indx = np.argsort(-neg_overlap)[:n_neg_mols]
        negStructure_smiles_ex = [training_smiles[x] for x in top_neg_indx]
        
        for smarts_idx in range(len(negSMARTS)):
            smarts = negSMARTS[smarts_idx]
            match_index = get_SMARTS_match_index(smarts, training_smiles)
            
            while True:
                r_index = random.randint(0, len(match_index)-1)
                if (X_train[match_index[r_index], negSMARTS_indx[smarts_idx]] == 1) and (Chem.MolFromSmiles(training_smiles[match_index[r_index]]).HasSubstructMatch(Chem.MolFromSmarts(smarts))):
                    break

            example_mol = Chem.MolFromSmiles(training_smiles[match_index[r_index]])
            rdDepictor.Compute2DCoords(example_mol)
            rdDepictor.StraightenDepiction(example_mol)

            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fpg.GetFingerprint(example_mol, additionalOutput=ao)
            clause_bits.append((example_mol, negSMARTS_indx[smarts_idx], ao.GetBitInfoMap()))

            if smarts_idx == 0 and len(posSMARTS) == 0:
                clause_logic.append('NOT')
                
            else:
                clause_logic.append('AND NOT')
            
        for neg_smiles in negStructure_smiles_ex:
            neg_mol = Chem.MolFromSmiles(neg_smiles)   
            neg_atom_Ws = np.zeros(Descriptors.HeavyAtomCount(neg_mol))
            for smarts_idx in range(len(negSMARTS)):
                smarts = negSMARTS[smarts_idx]
                patt = Chem.MolFromSmarts(smarts)
                matches = neg_mol.GetSubstructMatches(patt)
                for match in matches:
                    for atom_idx in match:
                        neg_atom_Ws[atom_idx] -= W / tm.number_of_clauses / len(matches)
                
            d2d = Draw.MolDraw2DCairo(550,350)
            SimilarityMaps.GetSimilarityMapFromWeights(
                mol=neg_mol,
                weights=list(neg_atom_Ws),
                draw2d=d2d,
                alpha=0,
                contourLines=10
            )
            
            d2d.FinishDrawing()
            mol_imgs.append(
                show_mol(d2d=d2d)
            )    
            
    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    drawOptions.prepareMolsBeforeDrawing = False
    drawOptions.legendFontSize = 40
    drawOptions.fixedFontSize = 20
    clause_image = Draw.DrawMorganBits(
        clause_bits, 
        molsPerRow=bits_per_row, 
        legends=clause_logic,
        drawOptions=drawOptions
    )
    return clause_image, show_images(mol_imgs, nCols=3, buffer=0)

def retrieve_literal_smarts(literal_indxs, X_train, training_smiles, fpg):
    smart_lst = []
    if literal_indxs is None or len(literal_indxs) == 0:
        return smart_lst
    for literal_indx in literal_indxs:
        pulled_indx = literal_indx
        literal_smiles = [training_smiles[i] for i in np.argwhere(X_train[:, pulled_indx]==1).flatten()]
        training_smarts_lst= []
        for smi in literal_smiles:
            mol = Chem.MolFromSmiles(smi)
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            fpg.GetFingerprint(mol, additionalOutput=ao)
            bitInfoMap = ao.GetBitInfoMap()
            aid, rad = bitInfoMap[pulled_indx][0]
            _smi, smart = getSubstructureSmi(mol,aid,rad)
            training_smarts_lst.append(smart.strip())
       
        majority_smart = Counter(training_smarts_lst).most_common()[0][0]
        smart_lst.append(majority_smart)
    return smart_lst


def WAC_literal_logic(feature_indx, query_smiles, fpg):
    # return " " if smarts is found in structure of all query smiles
    # return "!" if smarts is found in none of the query smiles
    # return f"{proportion}" where the variable "proportion" is the fraction of smarts matches in the query smiles
    query_mols = [Chem.MolFromSmiles(smi) for smi in query_smiles]
    
    query_fps = fpg.GetFingerprints(query_mols)
    query_fp_arr = np.array([fp_to_np(fp) for fp in query_fps], dtype=np.uint32)
    
    legend_lst = []
    logic_vals = []
    for featureIndx in feature_indx:
        proportion = round(np.sum(query_fp_arr[:, featureIndx]) / len(query_smiles), 1)
        legend_lst.append(f"{proportion}")
        logic_vals.append(proportion)
            
    return legend_lst, logic_vals


def get_SMARTS_match_index(smarts, training_smiles):
    match_index = []
    smarts_mol = Chem.MolFromSmarts(smarts)
    for i, smi in enumerate(training_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol.HasSubstructMatch(smarts_mol):
            match_index.append(i)
    return match_index

def WAC_plot(query_smiles, X_train, training_smiles, tm, max_literals=10, figsize=(10,8), zoom=0.5, fpg=None, negStructures=False):
    query_mols = [Chem.MolFromSmiles(smi) for smi in query_smiles]
    
    query_fps = fpg.GetFingerprints(query_mols)
    query_fp_arr = np.array([fp_to_np(fp) for fp in query_fps], dtype=np.uint32)
    
    active_W =tm.get_weights(1) 
    inactive_W = tm.get_weights(0) 
    
    A_query = tm.transform(query_fp_arr)
    print("A_query shape:", A_query.shape)
    print("A_query Sum", np.sum(A_query))
    
    S = np.array([[tm.get_ta_state(clause, ta) for ta in range(tm.clause_bank.number_of_literals)] for clause in range(tm.number_of_clauses)])
  
    C = (S > 127).astype(int)
   
    Cpos, Cneg = C[:, :C.shape[1]//2], C[:, C.shape[1]//2:]
    
    # Cneg - good in terms of global WAC / WC
    # General negations
    if negStructures:
        active_WAC = (active_W * A_query).sum(axis=0) @ (Cpos - Cneg)
        inactive_WAC = (inactive_W * A_query).sum(axis=0) @ (Cpos - Cneg)
    else:
        active_WAC = (active_W * A_query).sum(axis=0) @ (Cpos)
        inactive_WAC = (inactive_W * A_query).sum(axis=0) @ (Cpos)
    
    # Cpos only
    # Cpos - Cneg
    
    # active_WAC - inactive_WAC
    
    WAC = active_WAC + inactive_WAC

    # -WAC sort
    # +WAC
    
    feature_indx = np.argsort(-WAC)[:max_literals]
    print("Feature index", feature_indx)
    feature_importance = (WAC[feature_indx] / tm.number_of_clauses) / len(query_smiles)
    imp_max = np.max(np.abs(feature_importance))
    indx_str = feature_indx.astype(str).tolist()
    smarts = retrieve_literal_smarts(feature_indx, X_train=X_train, training_smiles=training_smiles, fpg=fpg)
    literal_logic, logic_vals = WAC_literal_logic(feature_indx, query_smiles, fpg)
    
    fig, axs = plt.subplots(1,1, figsize=figsize)
    
    pos_imp = np.clip(feature_importance, a_min=0, a_max=None)
    neg_imp = np.clip(feature_importance, a_min=None, a_max=0)
    
    cmap = plt.get_cmap('coolwarm')
    # invert logic values
    colors = cmap([1-x for x in logic_vals])
    axs.bar(indx_str, feature_importance, color=colors, align='center')
    axs.set_ylim(0, imp_max*1.1)
    
    # needs to be 
    labels = axs.xaxis.get_ticklabels()
    # label_texts = [label.get_text() for label in labels]

    # use drawmorganbits to draw smarts
    # need: the index of the bit, the smarts string, molecule example, ao map
    # get most negative structures
    for i in range(len(smarts)):
        match_index = get_SMARTS_match_index(smarts[i], training_smiles)
        match_index_len = len(match_index)
        
        # strange that we have to do this at all - but sometimes the match index is empty - minority bits?
        while True:
            r_index = random.randint(0, match_index_len-1)
            if (X_train[match_index[r_index], feature_indx[i]] == 1) and (Chem.MolFromSmiles(training_smiles[match_index[r_index]]).HasSubstructMatch(Chem.MolFromSmarts(smarts[i]))):
                break
        
        example_smi = training_smiles[match_index[r_index]]
        example_mol = Chem.MolFromSmiles(example_smi)
        
        rdDepictor.Compute2DCoords(example_mol)
        rdDepictor.StraightenDepiction(example_mol)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        fpg.GetFingerprint(example_mol, additionalOutput=ao)
        drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
        drawOptions.prepareMolsBeforeDrawing = False
        drawOptions.legendFontSize = 40
        drawOptions.fixedFontSize = 20
        pngimage = Draw.DrawMorganBit(
            mol=example_mol, 
            bitId=feature_indx[i], 
            bitInfo=ao.GetBitInfoMap(),
            legend=f"{literal_logic[i]}",
            drawOptions=drawOptions
        )
        bio = BytesIO()
        pngimage.save(bio, format='PNG')
        plt_img = plt.imread(bio, format='PNG')
        ib = OffsetImage(plt_img, zoom=zoom)
        ib.image.axes = axs
        
        # box alignment can adjust the position of the image relative to the tick
        ab = AnnotationBbox(ib, labels[i].get_position(), frameon=False, box_alignment=(0.5, 1.2))
        axs.add_artist(ab)
        
    axs.set_xticks([])
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axs.set_ylabel(ylabel="$WAC$ $mol^{-1}$ $clause^{-1}$", fontsize=24)
    axs.tick_params(axis='y', labelsize=24)
    plt.show() 
    
    return feature_importance, smarts


def tm_compound_map(query_smiles, tm, X_train, training_smiles, fpg):
    query_mol = Chem.MolFromSmiles(query_smiles)
    
    # get cleaned 2D coords
    rdDepictor.Compute2DCoords(query_mol)
    # get fingerprint depiction which lines up with tm training
    fps = fpg.GetFingerprints([query_mol])
    fps_np = np.array([fp_to_np(i) for i in fps], dtype=np.uint32)
    
    active_W = tm.get_weights(1)
    inactive_W = tm.get_weights(0)
    
   
    A_smiles = tm.transform(fps_np)
    print(A_smiles.shape)
    
    C_smiles_idx = np.argwhere(A_smiles[0]==1).flatten()
    
    S = np.array([[tm.get_ta_state(C_idx, ta) for ta in range(tm.clause_bank.number_of_literals)] for C_idx in C_smiles_idx], dtype=np.int16)

    Spos, Sneg = S[:, :S.shape[1]//2], S[:, S.shape[1]//2:]
    
    C = (S>127).astype(int)
    Cpos, Cneg = (Spos>127).astype(int), (Sneg>127).astype(int)

    query_atom_Ws = np.zeros(Descriptors.HeavyAtomCount(query_mol))
    
    # element-wise multiplication of inverse fingerprint and sum
    
    for clause_indx in range(C.shape[0]):
        # grab the weights
        # grab the smarts each with weight
        clause_active_W, clause_inactive_W = active_W[clause_indx], inactive_W[clause_indx]
        
        pos_subStructure_index, neg_subStructure_index = np.argwhere(Cpos[clause_indx] == 1), np.argwhere(Cneg[clause_indx] == 1)
        positive_smarts = retrieve_literal_smarts(pos_subStructure_index.flatten(), X_train, training_smiles, fpg=fpg)
        for smarts in positive_smarts:
            patt = Chem.MolFromSmarts(smarts)
            matches = query_mol.GetSubstructMatches(patt)
            for match in matches:
                for atom_idx in match:
                    query_atom_Ws[atom_idx] += ((clause_active_W + clause_inactive_W) / tm.number_of_clauses)  / len(matches)

    imgs = []
    d2d = Draw.MolDraw2DCairo(550, 350)
    dopts = d2d.drawOptions()
    SimilarityMaps.GetSimilarityMapFromWeights(mol=query_mol, weights=list(query_atom_Ws), draw2d=d2d, alpha=0, contourLines=10)
    d2d.FinishDrawing()
    imgs.append(
        show_mol(d2d=d2d)
    )

    return show_images(imgs, nCols=1)