import pandas as pd
import os
import requests
from urllib.parse import urlparse
import argparse

def download_image(image_url, image_id, download_dir):
    """
    Descarga una imagen desde una URL y la guarda localmente.
    
    Args:
        image_url: URL de la imagen
        image_id: ID único para el nombre del archivo
        download_dir: Directorio donde guardar la imagen
    
    Returns:
        str: Ruta local de la imagen descargada, o None si falla
    """
    try:
        # Crear directorio si no existe
        os.makedirs(download_dir, exist_ok=True)
        
        # Obtener extensión del archivo desde la URL
        parsed_url = urlparse(image_url)
        file_ext = os.path.splitext(parsed_url.path)[1]
        if not file_ext:
            file_ext = '.jpg'  # Extensión por defecto
        
        # Nombre del archivo local
        local_filename = f"{image_id}{file_ext}"
        local_path = os.path.join(download_dir, local_filename)
        
        # Descargar imagen
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Guardar imagen
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Imagen descargada: {local_filename}")
        return local_filename#local_path
        
    except Exception as e:
        print(f"✗ Error descargando {image_url}: {str(e)}")
        return None

def process_tsv_to_dataset(tsv_file, output_csv, download_dir="images", sample_size=None):
    """
    Procesa un archivo TSV y lo convierte al formato del MultimodalDataset.
    
    Args:
        tsv_file: Ruta al archivo TSV de entrada
        output_csv: Ruta al archivo CSV de salida
        download_dir: Directorio para guardar imágenes descargadas
        sample_size: Número de muestras a procesar (None para todas)
    """
    
    # Leer archivo TSV
    print(f"Leyendo archivo TSV: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Mostrar información del dataset
    print(f"Filas totales: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    
    # Mostrar distribución de etiquetas
    if '2_way_label' in df.columns:
        print("\nDistribución de etiquetas (2_way_label):")
        print(df['2_way_label'].value_counts().sort_index())
    
    # Tomar muestra si se especifica
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"\nMuestra de {sample_size} filas seleccionada")
    
    # Preparar datos para el nuevo CSV
    processed_data = []
    download_count = 0
    skip_count = 0
    
    print(f"\nProcesando imágenes...")
    
    for idx, row in df.iterrows():
        # Verificar que tenga imagen y texto
        if pd.isna(row.get('image_url')) or pd.isna(row.get('title')):
            skip_count += 1
            continue
            
        # Descargar imagen
        image_path = download_image(
            image_url=row['image_url'],
            image_id=row['id'],
            download_dir=download_dir
        )
        
        if image_path:
            download_count += 1
            
            # Crear entrada para el dataset
            processed_row = {
                'image_path': image_path,
                'text_content': str(row['title']),  # Usar 'title' como texto
                'label': int(row['2_way_label']),    # Usar '2_way_label' como etiqueta               


                'author': row['author'], 
                'clean_title': row['clean_title'],  
                'created_utc': row['created_utc'], 
                'domain': row['domain'], 
                'hasImage': row['hasImage'], 
                #'id': row['id'], 
                #'image_url': row[''], 
                'linked_submission_id': row['linked_submission_id'], 
                'num_comments': row['num_comments'], 
                'score': row['score'], 
                'subreddit': row['subreddit'], 
                'title': row['title'], 
                'upvote_ratio': row['upvote_ratio'], 
                '2_way_label': row['2_way_label'], 
                '3_way_label': row['3_way_label'], 
                '6_way_label': row['6_way_label'],
                 'image_url': row['image_url'],
                'image_id': row['id'],

            }
            processed_data.append(processed_row)
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Procesadas {idx + 1} filas...")
    
    # Crear DataFrame procesado
    processed_df = pd.DataFrame(processed_data)
    
    # Guardar CSV procesado
    processed_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Procesamiento completado!")
    print(f"📊 Estadísticas:")
    print(f"   - Total filas originales: {len(df)}")
    print(f"   - Imágenes descargadas exitosamente: {download_count}")
    print(f"   - Filas omitidas (sin imagen/texto): {skip_count}")
    print(f"   - Filas en CSV final: {len(processed_df)}")
    print(f"   - Archivo CSV guardado en: {output_csv}")
    print(f"   - Imágenes guardadas en: {download_dir}")
    
    # Mostrar distribución de etiquetas en el dataset final
    if len(processed_df) > 0:
        print(f"\n📈 Distribución de etiquetas en dataset final:")
        label_counts = processed_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"   - Label {label}: {count} muestras ({count/len(processed_df)*100:.1f}%)")

def create_sample_dataset(tsv_file, output_csv, sample_size=100):
    """
    Crea un dataset de muestra pequeño para pruebas.
    """
    process_tsv_to_dataset(tsv_file, output_csv, "sample_images", sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesar TSV para MultimodalDataset')
    parser.add_argument('--tsv_file', type=str, required=True, help='Ruta al archivo TSV de entrada')
    parser.add_argument('--output_csv', type=str, default='multimodal_dataset.csv', help='Ruta al CSV de salida')
    parser.add_argument('--download_dir', type=str, default='images', help='Directorio para imágenes descargadas')
    parser.add_argument('--sample_size', type=int, help='Tamaño de muestra (opcional)')
    
    args = parser.parse_args()
    
    # Instalar dependencias si no están disponibles
    try:
        import requests
    except ImportError:
        print("Instalando dependencias faltantes...")
        os.system("pip install requests pandas")
    
    # Procesar el dataset
    process_tsv_to_dataset(
        tsv_file=args.tsv_file,
        output_csv=args.output_csv,
        download_dir=args.download_dir,
        sample_size=args.sample_size
    )