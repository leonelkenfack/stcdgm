"""
Script pour nettoyer les lignes répétées dans un notebook Jupyter.
"""
import json
import sys
from pathlib import Path

def clean_repeated_lines(notebook_path: Path, max_repeats: int = 3):
    """
    Nettoie les lignes répétées consécutives dans un notebook.
    
    Parameters
    ----------
    notebook_path : Path
        Chemin vers le notebook à nettoyer
    max_repeats : int
        Nombre maximum de répétitions consécutives autorisées
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"Fichier non trouve: {notebook_path}")
        return
    
    print(f"Lecture du notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    total_removed = 0
    cells_modified = 0
    
    for cell_idx, cell in enumerate(nb['cells']):
        if 'source' not in cell:
            continue
        
        source = cell['source']
        if isinstance(source, str):
            source = source.split('\n')
            was_string = True
        else:
            was_string = False
        
        # Supprimer les lignes répétées consécutives
        cleaned_source = []
        prev_line = None
        repeat_count = 0
        
        for line in source:
            # Ignorer les lignes vides dans le comptage de répétitions
            line_stripped = line.strip()
            
            if line_stripped == prev_line and line_stripped:  # Ignorer les lignes vides
                repeat_count += 1
                if repeat_count <= max_repeats:
                    cleaned_source.append(line)
                else:
                    total_removed += 1
            else:
                repeat_count = 1
                cleaned_source.append(line)
                if line_stripped:
                    prev_line = line_stripped
        
        # Restaurer le format original
        if was_string:
            cleaned_source = '\n'.join(cleaned_source)
        else:
            # Pour les listes, garder les nouvelles lignes dans les chaînes
            cleaned_source = [line if line.endswith('\n') or i == len(cleaned_source) - 1 
                             else line + '\n' if not line.endswith('\n') else line
                             for i, line in enumerate(cleaned_source)]
        
        if cleaned_source != source:
            cell['source'] = cleaned_source
            cells_modified += 1
            print(f"  Cellule {cell_idx}: {len(source) - len(cleaned_source)} lignes repetees supprimees")
    
    if cells_modified > 0:
        # Créer une sauvegarde
        backup_path = notebook_path.with_suffix('.ipynb.bak')
        print(f"Sauvegarde creee: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        # Sauvegarder le fichier nettoyé
        print(f"Sauvegarde du fichier nettoye: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        print(f"\nNettoyage termine!")
        print(f"   - {cells_modified} cellule(s) modifiee(s)")
        print(f"   - {total_removed} ligne(s) repetee(s) supprimee(s)")
    else:
        print("\nAucune ligne repetee trouvee.")

if __name__ == "__main__":
    notebook_path = Path("../st_cdgm_training_evaluation.ipynb")
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])
    
    clean_repeated_lines(notebook_path, max_repeats=1)  # Garder seulement 1 occurrence

