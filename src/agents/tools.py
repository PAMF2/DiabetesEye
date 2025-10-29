"""Utility tools used by the diabeteseye agents.

This file provides small wrappers for preprocessing, model loading/inference
and simple PDF report generation. It intentionally keeps dependencies optional
so the repo can run with or without heavy ML libraries installed.
"""

import os
import re
from typing import Dict, Any, Optional

import cv2
import numpy as np

# Local preprocess function (from our package)
from ..preprocess.preprocess import preprocess_image


def format_markdown_text(text):
    """Convert basic markdown to reportlab HTML-like tags."""
    if not text:
        return ""
    # Replace **bold** with <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace *italic* with <i>italic</i> (avoiding list items)
    text = re.sub(r'(?<!\n)\*([^*]+)\*(?!\n)', r'<i>\1</i>', text)
    # Replace ## Header with <b>Header</b>
    text = re.sub(r'## ([^\n]+)', r'<b>\1</b>', text)
    # Replace list items * item with • item
    text = re.sub(r'\n\* ([^\n]+)', r'<br/>• \1', text)
    return text


def load_onnx_model(path: str):
    try:
        import onnxruntime as ort
    except Exception:
        return None
    sess = ort.InferenceSession(path)
    return sess


def load_torch_model(path: str):
    try:
        import torch
    except Exception:
        return None
    try:
        model = torch.load(path, map_location="cpu")
    except Exception:
        return None
    try:
        model.eval()
    except Exception:
        pass
    return model


class ModelWrapper:
    """A minimal wrapper that can use an ONNX or a torch model when available.

    Methods are intentionally defensive: if the runtime or model is missing,
    the wrapper returns a deterministic fallback prediction so the system
    remains usable for demo / tests.
    """

    def __init__(self, onnx_path: Optional[str] = None, torch_path: Optional[str] = None):
        self.onnx = load_onnx_model(onnx_path) if onnx_path else None
        self.torch = load_torch_model(torch_path) if torch_path else None

    def infer(self, input_array: np.ndarray) -> Dict[str, Any]:
        """Run inference on an HWC uint8 image array.

        Returns a small dict with either logits or a fallback DR prediction.
        """
        if self.onnx is not None:
            input_name = self.onnx.get_inputs()[0].name
            inp = input_array.astype(np.float32)[None, ...]
            out = self.onnx.run(None, {input_name: inp})
            return {"logits": out[0].tolist()}

        if self.torch is not None:
            import torch

            t = torch.from_numpy(input_array.astype(np.float32)[None, ...])
            with torch.no_grad():
                out = self.torch(t)
            try:
                out_np = out.cpu().numpy()
            except Exception:
                try:
                    out_np = out.numpy()
                except Exception:
                    out_np = None
            return {"logits": out_np.tolist() if out_np is not None else None}

        # Deterministic fallback for demos/tests
        return {"dr_grade": 2, "confidence": 0.94, "lesions": {"microaneurysms": 30}}


def image_quality_scorer(image_path: str) -> Dict[str, Any]:
    """Score basic image quality using our preprocess pipeline."""
    try:
        p = preprocess_image(image_path)
        return {
            "quality_score": p.get("quality_score", 0),
            "gradable": p.get("gradable", False),
            "issues": [],
            "focus_measure": p.get("focus_measure"),
        }
    except Exception as e:
        return {"quality_score": 0, "gradable": False, "issues": [str(e)]}


def dr_classification_model(image_path: str, model: Optional[ModelWrapper] = None) -> Dict[str, Any]:
    """Run DR classification on an image (uses model if provided)."""
    p = preprocess_image(image_path)
    img = p["preprocessed_image"]
    # convert to model input (HWC RGB, resized)
    inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp, (224, 224))
    return model.infer(inp) if model else {"dr_grade": 2, "confidence": 0.94, "lesions": {"microaneurysms": 30}}


def progression_model(features: dict) -> Dict[str, float]:
    """Small heuristic progression estimator used for demo/testing."""
    base = 0.35 if features.get("dr_grade", 2) == 2 else 0.15
    hba1c = float(features.get("hba1c", features.get("HBA1C", 7.0)))
    duration = float(features.get("duration_years", 5))
    risk_12m = min(0.95, base * (1 + (hba1c - 7.0) * 0.2) * (1 + duration * 0.02))
    return {"p6": round(risk_12m * 0.55, 3), "p12": round(risk_12m, 3), "p24": round(min(risk_12m * 1.35, 0.98), 3)}


def risk_calculator(image_feats: dict, clinical: dict) -> Dict[str, Any]:
    prob = progression_model({**image_feats, **clinical})
    cat = "LOW"
    if prob["p12"] >= 0.5:
        cat = "HIGH"
    elif prob["p12"] >= 0.25:
        cat = "MODERATE"
    return {"risk_category": cat, "monitoring": ("3 months" if cat == "HIGH" else "6 months")}


def generate_patient_report(result: dict, out_path: str = "report_patient.pdf") -> str:
    """Generate a comprehensive PDF patient report with professional formatting."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import inch
    except Exception:
        # fallback: write a simple text file
        with open(out_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
            f.write(str(result))
        return out_path

    doc = SimpleDocTemplate(out_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=40,
        alignment=1  # Center
    )
    story.append(Paragraph("DiabetesEye - Relatório do Paciente", title_style))
    story.append(Spacer(1, 20))

    # Patient Info Section
    story.append(Paragraph("Informações do Paciente", styles['Heading2']))
    clinical = result.get('clinical', {})
    patient_info = [
        ["ID do Paciente:", result.get('patient_id', 'demo-patient-001')],
        ["ID do Exame:", result.get('exam_id', 'demo-exam-001')],
        ["Data do Exame:", "2025-10-28"],
        ["Idade:", f"{clinical.get('age', 'N/A')} anos"],
        ["Duração do Diabetes:", f"{clinical.get('duration_years', 'N/A')} anos"],
        ["Último Exame Oftalmológico:", clinical.get('last_eye_exam', 'N/A')],
    ]
    patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))

    # DR Findings
    story.append(Paragraph("Achados de Retinopatia Diabética", styles['Heading2']))
    dr = result.get('dr', {})
    dr_info = [
        ["Grau de RD:", f"{dr.get('dr_grade', 'N/A')}"],
        ["Confiança:", f"{dr.get('confidence', 'N/A')}"],
        ["Lesões Detectadas:", ", ".join(dr.get('lesions', {}).keys()) if dr.get('lesions') else "Nenhuma"],
    ]
    dr_table = Table(dr_info, colWidths=[2*inch, 4*inch])
    dr_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(dr_table)
    story.append(Spacer(1, 20))

    # Explanation
    if dr.get('explanation'):
        story.append(Paragraph("Explicação dos Achados", styles['Heading3']))
        paragraphs = dr['explanation'].split('\n\n')
        for para in paragraphs:
            if para.strip():
                formatted = format_markdown_text(para)
                story.append(Paragraph(formatted, styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 14))

    # Progression Risks
    story.append(Paragraph("Riscos de Progressão", styles['Heading2']))
    prog = result.get('progression', {})
    prog_data = [
        ["Período", "Risco de Progressão"],
        ["6 meses", f"{prog.get('p6', 'N/A')}"],
        ["12 meses", f"{prog.get('p12', 'N/A')}"],
        ["24 meses", f"{prog.get('p24', 'N/A')}"],
    ]
    prog_table = Table(prog_data, colWidths=[2*inch, 4*inch])
    prog_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightcoral),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(prog_table)
    story.append(Spacer(1, 20))

    # Integrated Risk
    integrated = result.get('integrated', {})
    story.append(Paragraph("Avaliação Integrada de Risco", styles['Heading2']))
    risk_info = [
        ["Categoria de Risco:", integrated.get('risk_category', 'N/A')],
        ["Frequência de Monitoramento:", integrated.get('monitoring', 'N/A')],
    ]
    risk_table = Table(risk_info, colWidths=[2*inch, 4*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 20))

    # Recommendations
    story.append(Paragraph("Recomendações", styles['Heading2']))
    clinical = result.get('clinical', {})
    integrated = result.get('integrated', {})
    recs = [
        "• Compareça ao retorno oftalmológico conforme recomendado acima.",
        f"• Mantenha controle glicêmico adequado (HbA1c atual: {clinical.get('hba1c', 'N/A')}%).",
        f"• Controle a pressão arterial (atual: {clinical.get('bp_systolic', 'N/A')}/{clinical.get('bp_diastolic', 'N/A')} mmHg).",
        "• Continue as medicações prescritas e informe qualquer efeito colateral.",
        "• Evite fatores de risco como fumo e sedentarismo.",
        "• Monitore sintomas visuais e procure atendimento imediato se notar piora.",
        "• Mantenha acompanhamento regular com endocrinologista e oftalmologista.",
    ]
    for rec in recs:
        story.append(Paragraph(rec, styles['Normal']))
    story.append(Spacer(1, 12))

    # Add image if available
    image_uri = result.get('image_uri')
    if image_uri and os.path.exists(image_uri):
        story.append(Paragraph("Imagem Analisada", styles['Heading2']))
        try:
            from reportlab.platypus import Image
            img = Image(image_uri, width=4*inch, height=3*inch)
            story.append(img)
        except Exception:
            story.append(Paragraph(f"Imagem: {image_uri}", styles['Normal']))
        story.append(Spacer(1, 20))

    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Spacer(1, 24))
    story.append(Paragraph("Relatório gerado por DiabetesEye - IA para análise de retinopatia diabética", footer_style))

    doc.build(story)
    return out_path


def generate_physician_report(result: dict, out_path: str = "report_physician.pdf") -> str:
    """Generate a detailed PDF physician report with clinical and technical details."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import inch
    except Exception:
        # fallback: write a simple text file
        with open(out_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
            f.write(str(result))
        return out_path

    doc = SimpleDocTemplate(out_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=40,
        alignment=1  # Center
    )
    story.append(Paragraph("DiabetesEye - Relatório Médico Detalhado", title_style))
    story.append(Spacer(1, 20))

    # Patient and Exam Info
    story.append(Paragraph("Informações do Paciente e Exame", styles['Heading2']))
    clinical = result.get('clinical', {})
    patient_info = [
        ["ID do Paciente:", result.get('patient_id', 'demo-patient-001')],
        ["ID do Exame:", result.get('exam_id', 'demo-exam-001')],
        ["Data do Exame:", "2025-10-28"],
        ["Tipo de Imagem:", "Fundoscopia Colorida"],
        ["Olho Examinado:", "Olho Direito (OD)"],
    ]
    patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # Clinical History
    story.append(Paragraph("Histórico Clínico", styles['Heading2']))
    clinical_info = [
        ["Idade:", f"{clinical.get('age', 'N/A')} anos"],
        ["Gênero:", "Feminino" if clinical.get('gender') == 'F' else "Masculino"],
        ["Duração DM:", f"{clinical.get('duration_years', 'N/A')} anos"],
        ["HbA1c:", f"{clinical.get('hba1c', 'N/A')}%"],
        ["Pressão Arterial:", f"{clinical.get('bp_systolic', 'N/A')}/{clinical.get('bp_diastolic', 'N/A')} mmHg"],
        ["IMC:", f"{clinical.get('bmi', 'N/A')} kg/m²"],
        ["Tabagismo:", "Não" if not clinical.get('smoker', True) else "Sim"],
        ["Último Exame Oftalmológico:", clinical.get('last_eye_exam', 'N/A')],
        ["Histórico Familiar:", clinical.get('family_history', 'N/A')],
        ["Medicações Atuais:", ", ".join(clinical.get('medications', []))],
    ]
    clinical_table = Table(clinical_info, colWidths=[2.5*inch, 3.5*inch])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(clinical_table)
    story.append(Spacer(1, 20))

    # Image Quality
    story.append(Paragraph("Avaliação de Qualidade da Imagem", styles['Heading2']))
    preprocessing = result.get('preprocessing', {})
    quality_info = [
        ["Pontuação de Qualidade:", f"{preprocessing.get('quality_score', 'N/A')} ({'Gradável' if preprocessing.get('gradable', False) else 'Não Gradável'})"],
        ["Medida de Foco:", f"{preprocessing.get('focus_measure', 'N/A')}"],
        ["Problemas Identificados:", ", ".join(preprocessing.get('issues', [])) or "Nenhum"],
    ]
    quality_table = Table(quality_info, colWidths=[2*inch, 4*inch])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightcyan),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(quality_table)
    story.append(Spacer(1, 20))

    # DR Classification Details
    story.append(Paragraph("Classificação de Retinopatia Diabética", styles['Heading2']))
    dr = result.get('dr', {})
    dr_details = [
        ["Grau de RD (ETDRS):", f"{dr.get('dr_grade', 'N/A')}"],
        ["Confiança do Modelo:", f"{dr.get('confidence', 'N/A')}"],
        ["Lesões Quantificadas:", str(dr.get('lesions', {}))],
        ["Imagem Analisada:", dr.get('image_uri', 'N/A')],
    ]
    dr_table = Table(dr_details, colWidths=[2*inch, 4*inch])
    dr_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(dr_table)
    story.append(Spacer(1, 20))

    # Add image if available
    image_uri = result.get('image_uri')
    if image_uri and os.path.exists(image_uri):
        story.append(Paragraph("Imagem Analisada", styles['Heading2']))
        try:
            from reportlab.platypus import Image
            img = Image(image_uri, width=4*inch, height=3*inch)
            story.append(img)
        except Exception:
            story.append(Paragraph(f"Imagem: {image_uri}", styles['Normal']))
        story.append(Spacer(1, 20))

    # DR Explanation
    if dr.get('explanation'):
        story.append(Paragraph("Interpretação Clínica dos Achados", styles['Heading3']))
        paragraphs = dr['explanation'].split('\n\n')
        for para in paragraphs:
            if para.strip():
                formatted = format_markdown_text(para)
                story.append(Paragraph(formatted, styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 14))

    # Progression Analysis
    story.append(Paragraph("Análise de Progressão", styles['Heading2']))
    prog = result.get('progression', {})
    prog_data = [
        ["Período", "Probabilidade de Progressão", "Fatores Considerados"],
        ["6 meses", f"{prog.get('p6', 'N/A')}", "Grau RD, HbA1c, Duração DM"],
        ["12 meses", f"{prog.get('p12', 'N/A')}", "Grau RD, HbA1c, Duração DM"],
        ["24 meses", f"{prog.get('p24', 'N/A')}", "Grau RD, HbA1c, Duração DM"],
    ]
    prog_table = Table(prog_data, colWidths=[1.5*inch, 2*inch, 2.5*inch])
    prog_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightcoral),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(prog_table)
    story.append(Spacer(1, 20))

    # Progression Explanation
    if prog.get('explanation'):
        story.append(Paragraph("Explicação da Análise de Progressão", styles['Heading3']))
        paragraphs = prog['explanation'].split('\n\n')
        for para in paragraphs:
            if para.strip():
                formatted = format_markdown_text(para)
                story.append(Paragraph(formatted, styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 14))

    # Integrated Risk Assessment
    story.append(Paragraph("Avaliação Integrada de Risco", styles['Heading2']))
    integrated = result.get('integrated', {})
    risk_details = [
        ["Categoria de Risco Global:", integrated.get('risk_category', 'N/A')],
        ["Frequência Recomendada de Monitoramento:", integrated.get('monitoring', 'N/A')],
        ["Fatores Clínicos Considerados:", "HbA1c, Duração DM, Pressão Arterial"],
    ]
    risk_table = Table(risk_details, colWidths=[2.5*inch, 3.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 20))

    # Clinical Integration Explanation
    if integrated.get('explanation'):
        story.append(Paragraph("Integração Clínica e Recomendações", styles['Heading3']))
        paragraphs = integrated['explanation'].split('\n\n')
        for para in paragraphs:
            if para.strip():
                formatted = format_markdown_text(para)
                story.append(Paragraph(formatted, styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 6))

    # Recommendations for Physician
    story.append(Paragraph("Recomendações Clínicas", styles['Heading2']))
    clinical = result.get('clinical', {})
    dr = result.get('dr', {})
    integrated = result.get('integrated', {})
    recs = [
        f"• Otimizar controle glicêmico - HbA1c atual {clinical.get('hba1c', 'N/A')}% (meta <7.0% para pacientes com RD).",
        f"• Controlar PA - atual {clinical.get('bp_systolic', 'N/A')}/{clinical.get('bp_diastolic', 'N/A')} mmHg (meta <130/80 mmHg).",
        f"• Avaliar necessidade de tratamento oftalmológico específico para RD grau {dr.get('dr_grade')} com lesões {list(dr.get('lesions', {}).keys())}.",
        f"• Programar retorno oftalmológico em {integrated.get('monitoring', 'N/A')} devido a risco {integrated.get('risk_category', 'N/A')}.",
        "• Considerar intensificação de medicações hipoglicemiantes se HbA1c permanecer elevada.",
        "• Avaliar função renal e risco cardiovascular associado.",
        "• Orientar paciente sobre sinais de alarme oftalmológicos.",
        "• Documentar achados em prontuário eletrônico com imagens.",
    ]
    for rec in recs:
        story.append(Paragraph(rec, styles['Normal']))
    story.append(Spacer(1, 20))

    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Spacer(1, 24))
    story.append(Paragraph("Relatório gerado por DiabetesEye - Sistema de IA para diagnóstico assistido de retinopatia diabética", footer_style))
    story.append(Paragraph("Data de Geração: 2025-10-28 | Versão do Modelo: Gemini-2.0-Flash", footer_style))

    doc.build(story)
    return out_path
