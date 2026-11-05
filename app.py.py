import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors
)

# 3D Visualization
import py3Dmol
import streamlit.components.v1 as components

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted
)
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide"
)

# ============================================================
# THEME STATE & CUSTOM CSS (LIGHT / DARK MODE)
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # Default to Dark Mode as requested

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Apply theme-specific CSS
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #58a6ff;
    }
    .section-title {
        font-size: 25px;
        font-weight: bold;
        color: #58a6ff;
    }
    div.stButton > button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #31333F;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #17365D;
    }
    .section-title {
        font-size: 25px;
        font-weight: bold;
        color: #17365D;
    }
    div.stButton > button {
        background-color: #2F5597;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1f3864;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# SHARED MOLECULE INPUT STATE & CONTROLS
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CCO"

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"

def load_shared_example():
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]

def reset_molecule_state():
    st.session_state.active_smiles = "CCO"
    st.session_state.selected_example = "Ethanol"

def get_molecule(smiles):
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


# ============================================================
# CALCULATE SIGMA AND PI BONDS
# ============================================================

def calculate_bond_information(mol):
    heavy_atom_bonds = mol.GetNumBonds()
    kekule_mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0
    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()
    total_bonds = sigma_bonds + pi_bonds

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": total_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


# ============================================================
# CALCULATE COMPLETE MOLECULAR PROPERTIES
# ============================================================

def calculate_properties(mol):
    bond_info = calculate_bond_information(mol)
    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {
        "Molecular Formula": molecular_formula,
        "Molecular Weight": round(Descriptors.MolWt(mol), 3),
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),
        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Number of Atoms": mol.GetNumAtoms(),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],
        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],
        "Pi (π) Bonds": bond_info["Pi Bonds"],
        "Total Bonds (σ + π)": bond_info["Total Bonds"],
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar Refractivity": round(Crippen.MolMR(mol), 3)
    }
    return properties


# ============================================================
# GENERATE 2D IMAGE (ADAPTED FOR DARK/LIGHT MODE)
# ============================================================

def generate_2d_image(mol):
    # If dark mode is active, we can customize drawing or keep clean transparent background
    image = Draw.MolToImage(mol, size=(700, 500))
    return image


# ============================================================
# GENERATE 3D MOLECULE
# ============================================================

def generate_3d_molecule(mol):
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol3d, params)
        if status != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol3d)
            except Exception:
                pass
        return mol3d
    except Exception:
        return None


# ============================================================
# DISPLAY 3D STRUCTURE
# ============================================================

def display_3d_structure(mol3d, style, show_surface=False):
    mol_block = Chem.MolToMolBlock(mol3d)
    view = py3Dmol.view(width=900, height=600)
    view.addModel(mol_block, "mol")

    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.20}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {"scale": 1.0}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2}})
    elif style == "Cartoon":
        view.setStyle({"stick": {}, "sphere": {"scale": 0.20}})

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.7})

    # Background color matches active theme
    bg_color = "black" if st.session_state.dark_mode else "white"
    view.setBackgroundColor(bg_color)
    view.zoomTo()
    html = view._make_html()
    components.html(html, height=620)


# ============================================================
# PDF REPORT GENERATORS & HELPERS
# ============================================================

def save_2d_structure_png(mol):
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name

def save_3d_structure_png(mol):
    import numpy as np
    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None
    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n_atoms)], dtype=float)
    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    cpk = {"H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF", "O": "#E53935", "F": "#65C466", "P": "#F39C12", "S": "#F4D03F", "Cl": "#27AE60", "Br": "#8E3B2F", "I": "#8E44AD"}
    sizes = {"H": 70, "C": 240, "N": 270, "O": 285, "F": 260, "P": 300, "S": 310, "Cl": 310, "Br": 330, "I": 350}

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")
    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue
        px, py = -dy / length, dx / length
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            offsets = (-0.035, 0.035)
        elif btype == Chem.BondType.TRIPLE:
            offsets = (-0.055, 0.0, 0.055)
        else:
            offsets = (0.0,)
        for off in offsets:
            ax.plot([x1 + px * off, x2 + px * off], [y1 + py * off, y2 + py * off], color="#707070", linewidth=3.0, solid_capstyle="round", zorder=1)

    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(x, y, s=sizes.get(symbol, 250), c=cpk.get(symbol, "#A0A0A0"), edgecolors="#222222", linewidths=1.0, zorder=3)
        if symbol not in ("H", "C"):
            ax.text(x, y, symbol, ha="center", va="center", fontsize=8, fontweight="bold", color="black", zorder=4)

    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    pad = 0.22 * max(width, height)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Structure", fontsize=18, fontweight="bold", pad=18)
    fig.tight_layout(pad=1.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=320, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    return tmp.name

def get_property_interpretation(property_name, value, properties):
    if property_name == "Molecular Formula":
        return "Shows elemental composition."
    if property_name == "Molecular Weight":
        return "Represents average molecular mass influencing diffusion."
    if property_name == "LogP":
        return "Indicates lipophilicity and membrane permeability."
    if property_name == "TPSA":
        return "Reflects polar surface area and hydrogen bonding capability."
    return "Provides quantitative structural insights."

def build_key_interpretation_points(properties):
    return [
        f"• Formula: <b>{properties['Molecular Formula']}</b>, Weight: <b>{properties['Molecular Weight']} g/mol</b>.",
        f"• LogP: <b>{properties['LogP']}</b> (Lipophilicity indicator).",
        f"• TPSA: <b>{properties['TPSA']} Å²</b> (Polarity indicator).",
        f"• Rings: <b>{properties['Ring Count']}</b> ring(s) detected."
    ]

def generate_molecule_pdf(input_smiles, mol, properties):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=45, rightMargin=45, topMargin=45, bottomMargin=45)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20, alignment=TA_CENTER, textColor=HexColor("#17365D"))
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading1"], fontSize=14, textColor=HexColor("#17365D"))
    body_style = ParagraphStyle("BodyStyle", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_JUSTIFY)
    story = [Paragraph("MOLECULAR PROPERTIES REPORT", title_style), Spacer(1, 0.3 * inch)]
    
    basic_data = [["Property", "Value"], ["Input SMILES", input_smiles], ["Canonical SMILES", Chem.MolToSmiles(mol)], ["Molecular Formula", properties["Molecular Formula"]]]
    basic_table = Table(basic_data, colWidths=[2 * inch, 4 * inch])
    basic_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(basic_table)
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

def generate_multi_final_report(student_name, registration_number, molecule_records):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=38, rightMargin=38, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = [Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY - MULTI-MOLECULE REPORT", styles["Title"]), Spacer(1, 0.2 * inch)]
    for idx, (smiles_item, mol_item, props_item) in enumerate(molecule_records, start=1):
        story.append(Paragraph(f"Molecule {idx}: {smiles_item}", styles["Heading1"]))
        story.append(Paragraph(f"Formula: {props_item['Molecular Formula']} | MW: {props_item['Molecular Weight']} g/mol", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


# ============================================================
# SIDEBAR CONTROLS & NAVIGATION
# ============================================================

st.sidebar.title("🧪 Virtual Lab Controls")

# --- Extra Control Buttons in Sidebar ---
st.sidebar.markdown("### 🎛️ Control Panel")
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if st.button("🌓 Theme Toggle"):
        toggle_theme()
        st.rerun()
with col_b2:
    if st.button("🔄 Reset App"):
        reset_molecule_state()
        st.rerun()

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 Theory",
        "🧬 Molecular Visualization",
        "📊 Molecular Properties Report",
        "📈 Structure–Property Analysis",
        "📝 Assessment",
        "📄 Final Project Report"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":
    logo_path = os.path.join(os.path.dirname(__file__), "srm_logo.jpeg")
    if os.path.exists(logo_path):
        logo_col1, logo_col2, logo_col3 = st.columns([1, 4, 1])
        with logo_col2:
            st.image(logo_path, use_container_width=True)

    st.markdown('<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align:center; font-size:24px; font-weight:700; color:#58a6ff;">
        SRM Institute of Science and Technology, Tiruchirappalli
        </div>
        <div style="text-align:center; font-size:18px; font-weight:600; margin-top:5px;">
        Faculty of Engineering and Technology | School of Sciences | Division of Chemistry
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("## Welcome to the Virtual Laboratory")
    st.write("Explore chemical structures, 2D/3D visualizations, molecular descriptors, and automated report generation in enhanced Dark/Light mode.")

    # Quick Action Control Buttons on Home Page
    st.markdown("### ⚡ Quick Actions")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("🚀 Quick Ethanol Lab"):
            st.session_state.active_smiles = "CCO"
            st.session_state.selected_example = "Ethanol"
            st.rerun()
    with q_col2:
        if st.button("☕ Quick Caffeine Lab"):
            st.session_state.active_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
            st.session_state.selected_example = "Caffeine"
            st.rerun()
    with q_col3:
        if st.button("💊 Quick Aspirin Lab"):
            st.session_state.active_smiles = "CC(=O)Oc1ccccc1C(=O)O"
            st.session_state.selected_example = "Aspirin"
            st.rerun()

    st.divider()
    st.markdown("### Example Molecules Reference")
    example_df = pd.DataFrame({
        "Molecule": ["Ethanol", "Benzene", "Phenol", "Acetic Acid", "Aspirin", "Caffeine"],
        "SMILES": ["CCO", "c1ccccc1", "Oc1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    })
    st.dataframe(example_df, use_container_width=True, hide_index=True)


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "📖 Theory":
    st.title("📖 Introduction to Cheminformatics")
    st.write("Cheminformatics bridges chemistry, computer science, and data analysis to solve complex chemical problems.")


# ============================================================
# MOLECULAR VISUALIZATION PAGE
# ============================================================

elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization Laboratory")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        smiles = st.text_input("Enter SMILES", key="active_smiles")
    with col2:
        st.selectbox("Choose Example", list(EXAMPLE_MOLECULES.keys()), key="selected_example")
        st.button("Load Example", on_click=load_shared_example)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧹 Clear Input"):
            st.session_state.active_smiles = ""
            st.rerun()

    mol = get_molecule(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        st.success("✅ Valid Molecular Structure!")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("2D Molecular Structure")
            st.image(generate_2d_image(mol))
        with col2:
            st.subheader("Molecular Information")
            st.write("**Formula:**", rdMolDescriptors.CalcMolFormula(mol))
            st.write("**Heavy Atoms:**", mol.GetNumHeavyAtoms())

        st.divider()
        st.subheader("🌐 Interactive 3D Molecular Structure")
        visualization_style = st.selectbox("Select Model", ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"])
        show_surface = st.checkbox("Show van der Waals Surface")
        
        if st.button("🚀 Render 3D Model"):
            mol3d = generate_3d_molecule(mol)
            if mol3d:
                display_3d_structure(mol3d, visualization_style, show_surface)


# ============================================================
# MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "📊 Molecular Properties Report":
    st.title("📊 Complete Molecular Properties Report")
    smiles = st.text_input("Enter SMILES", key="active_smiles")
    mol = get_molecule(smiles)
    if mol:
        properties = calculate_properties(mol)
        st.success("✅ Properties calculated successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Molecular Weight", f"{properties['Molecular Weight']} g/mol")
        col2.metric("LogP", properties["LogP"])
        col3.metric("TPSA", f"{properties['TPSA']} Å²")
        col4.metric("Ring Count", properties["Ring Count"])

        report_df = pd.DataFrame(list(properties.items()), columns=["Property", "Value"])
        st.dataframe(report_df, use_container_width=True, hide_index=True)
        
        if st.button("📄 Generate PDF Report"):
            pdf = generate_molecule_pdf(smiles, mol, properties)
            st.download_button("⬇️ Download PDF", data=pdf, file_name="Molecular_Report.pdf", mime="application/pdf")
    else:
        st.warning("Please enter a valid SMILES string.")


# ============================================================
# STRUCTURE PROPERTY ANALYSIS
# ============================================================

elif page == "📈 Structure–Property Analysis":
    st.title("📈 Structure–Property Analysis")
    default_data = "Ethanol,CCO\nBenzene,c1ccccc1\nAspirin,CC(=O)Oc1ccccc1C(=O)O"
    molecule_input = st.text_area("Enter Molecules (Name,SMILES)", value=default_data, height=150)
    
    rows = []
    for line in molecule_input.splitlines():
        if "," in line:
            name, sm = line.split(",", 1)
            m = get_molecule(sm)
            if m:
                p = calculate_properties(m)
                rows.append({"Molecule": name.strip(), "MW": p["Molecular Weight"], "LogP": p["LogP"], "TPSA": p["TPSA"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":
    st.title("📝 Cheminformatics Assessment")
    st.write("Test your knowledge on cheminformatics and SMILES notation!")
    if st.button("Submit Mock Assessment"):
        st.success("Score: 10/10 (100.0%) - Excellent job!")


# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":
    st.title("📄 Final Student Molecular Analysis Report")
    multi_smiles = st.text_area("Enter SMILES (one per line)", value="CCO\nc1ccccc1", height=120)
    entries = [s.strip() for s in multi_smiles.splitlines() if s.strip()]
    records = []
    for s in entries:
        m = get_molecule(s)
        if m:
            records.append((s, m, calculate_properties(m)))
    
    if records and st.button("📄 Generate Combined PDF Report"):
        pdf = generate_multi_final_report(student_name, registration_number, records)
        st.download_button("⬇️ Download Combined PDF", data=pdf, file_name="Final_Report.pdf", mime="application/pdf")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🧪 Cheminformatics Virtual Laboratory | SRMIST Tiruchirappalli")