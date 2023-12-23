import { useEffect, useContext, useState } from 'react';
import HomeContext from '@/pages/api/home/home.context';

export const libImages: { [key: string]: string } = {
  'scanpy': '/apps/scanpy.webp',
  'squidpy': '/apps/squidpy.webp',
  'pyteomics': '/apps/pyteomics.webp',
  'qiime2': '/apps/qiime2.webp',
  'scikit-bio': '/apps/scikitbio.webp',
  'biopython': '/apps/biopython.png',
  'biotite': '/apps/biotite.png',
  'deap': '/apps/deap.png',
  'eletoolkit': '/apps/eletoolkit.webp',
  'pyopenms': '/apps/pyopenms.webp',
  'scenicplus': '/apps/SCENIC.webp',
  'scvi-tools': '/apps/scvitools.webp',
  'sonata': '/apps/SONATA.webp',
  'MIOSTONE': '/apps/MIOSTONE.webp',
  'ehrapy': '/apps/ehrapy.webp',
  'custom': '/apps/customize.webp',
};

export const LibCardSelect = () => {
  const {
    state: { selectedConversation, methods },
    handleUpdateConversation,
    dispatch: homeDispatch,
  } = useContext(HomeContext);
  
  const [selectedLib, setSelectedLib] = useState<string | undefined>(selectedConversation?.Lib);
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [customLibName, setCustomLibName] = useState('');
  const [customGitHubURL, setCustomGitHubURL] = useState('');
  const [customReadTheDocsURL, setCustomReadTheDocsURL] = useState('');
  const [customAPIHTML, setcustomAPIHTML]  = useState('');
  const [customLIBALIAS, setcustomLIBALIAS]  = useState('');

  useEffect(() => {
    const newLibs = [
      { id: 'scanpy', name: 'scanpy' },
      { id: 'squidpy', name: 'squidpy' },
      { id: 'pyteomics', name: 'pyteomics' },
      { id: 'qiime2', name: 'qiime2' },
      { id: 'scikit-bio', name: 'scikit-bio' },
      { id: 'biopython', name: 'biopython' },
      { id: 'biotite', name: 'biotite' },
      { id: 'deap', name: 'deap' },
      { id: 'eletoolkit', name: 'eletoolkit' },
      { id: 'pyopenms', name: 'pyopenms' },
      { id: 'scenicplus', name: 'scenicplus' },
      { id: 'scvi-tools', name: 'scvi-tools' },
      { id: 'sonata', name: 'sonata' },
      { id: 'MIOSTONE', name: 'MIOSTONE' },
      { id: 'ehrapy', name: 'ehrapy' },
      { id: 'custom', name: 'custom' },
    ];
  
    const existingLibIds = methods.map(lib => lib.id);
    const uniqueNewLibs = newLibs.filter(lib => !existingLibIds.includes(lib.id));
  
    if (uniqueNewLibs.length > 0) {
      // homeDispatch({ field: 'methods', value: [...methods, ...uniqueNewLibs] });
      homeDispatch({ field: 'methods', value: newLibs });
    }
  }, [homeDispatch]);
  
  const handleCardClick = (lib: { id: string, name: string }) => {
    setSelectedLib(lib.id);
    setShowCustomInput(lib.id === 'custom');
    if (lib.id !== 'custom' && selectedConversation) {
      handleUpdateConversation(selectedConversation, {
        key: 'Lib',
        value: lib.id,
      });
    }
  };
  
  useEffect(() => {
    if (selectedLib === 'custom' && selectedConversation) {
      handleUpdateConversation(selectedConversation, {
        key: 'Lib',
        value: customLibName,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'new_lib_github_url',
        value: customGitHubURL,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'new_lib_doc_url',
        value: customReadTheDocsURL,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'api_html',
        value: customAPIHTML,
      });
      handleUpdateConversation(selectedConversation, {
        key: 'lib_alias',
        value: customLIBALIAS,
      });
    }
  }, [customLibName, customGitHubURL, customReadTheDocsURL, customAPIHTML, customLIBALIAS, selectedLib, selectedConversation, handleUpdateConversation]);
  return (
    <>
      <div className="select-container">
        <span className="text-[12px] text-black/50 dark:text-white/50 text-sm">
        Select the Lib
        </span>
      </div>
      <div className="card-select-container">
        {methods.map((lib) => (
          <div
            key={lib.id}
            className={`card ${selectedLib === lib.id ? 'card-selected' : ''}`}
            onClick={() => handleCardClick(lib)}
          >
            <img src={libImages[lib.id]} alt={lib.name} />
            <div>{lib.name}</div>
          </div>
        ))}
      </div>
      {showCustomInput && (
        <div className="custom-input-container">
          <div>
            <label style={{ color: 'black' }}>LIB Name: </label>
            <input 
              type="text" 
              placeholder="(Required) e.g. biopython" 
              style={{ color: 'black' }} 
              onChange={(e) => setCustomLibName(e.target.value)} 
            />
          </div>
          <div>
            <label style={{ color: 'black' }}>LIB ALIAS: </label>
            <input 
              type="text" 
              placeholder="(Required) e.g. Bio" 
              style={{ color: 'black' }} 
              onChange={(e) => setcustomLIBALIAS(e.target.value)} 
            />
          </div>
          <div>
            <label style={{ color: 'black' }}>API HTML: </label>
            <input 
              type="text" 
              placeholder="(Optional) e.g. biopython.org/docs/latest/api" 
              style={{ color: 'black' }} 
              onChange={(e) => setcustomAPIHTML(e.target.value)} 
            />
          </div>
          <div>
            <label style={{ color: 'black' }}>GitHub URL: </label>
            <input 
              type="text" 
              placeholder="(Optional) e.g. https://github.com/biopython/biopython" 
              style={{ color: 'black' }} 
              onChange={(e) => setCustomGitHubURL(e.target.value)}
            />
          </div>
          <div>
            <label style={{ color: 'black' }}>ReadTheDocs URL: </label>
            <input 
              type="text" 
              placeholder="(Optional) e.g. https://biopython.org/docs/latest/api" 
              style={{ color: 'black' }} 
              onChange={(e) => setCustomReadTheDocsURL(e.target.value)}
            />
          </div>
        </div>
      )}

      <style jsx>{`
        .select-container {
          display: flex;
          justify-content: flex-start;
          align-items: center;
          margin-bottom: 10px;
          font-family: 'Your Font Name', sans-serif; /* Replace 'Your Font Name' with the actual font name */
          font-size: 16px; /* Adjust font size as needed */
          font-weight: bold;
        }
        .card-select-container {
          display: flex;
          flex-wrap: wrap;
          align-items: flex-start;
          max-width: 800px;
          gap: 10px;
        }
        .card {
          cursor: pointer;
          width: calc(25% - 15px);
          height: 130px;
          margin-bottom: 10px;
          border: 1px solid gray;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          position: relative;
        }
        .card img {
          max-width: 100%;
          max-height: 80%;
          margin-bottom: 5px;
        }
        .card-selected {
          background-color: lightblue;
          color: #333333;
        }
        .card div {
          position: absolute;
          bottom: 0;
          left: 0;
          width: 100%;
          padding: 5px;
          background-color: rgba(255, 255, 255, 0.8);
        }
        .custom-input-container div {
          display: flex;
          align-items: center;
          margin-bottom: 20px;
        }
        label {
          flex-shrink: 0; 
          margin-right: 20px;
          white-space: nowrap; 
        }
        input {
          flex-grow: 1;
          min-width: 0;
        }
      `}</style>
    </>
  );
};